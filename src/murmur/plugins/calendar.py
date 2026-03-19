"""Murmur plugin: Google Calendar integration with multi-account support."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from murmur.config import get_section

console = Console()

CONFIG_DIR = Path.home() / ".config" / "murmur"
ACCOUNTS_DIR = CONFIG_DIR / "accounts"
CREDENTIALS_PATH = CONFIG_DIR / "google_credentials.json"
ACTIVE_ACCOUNT_PATH = CONFIG_DIR / "active_account"

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------


def _token_path(account):
    """Return the token path for a named account."""
    return ACCOUNTS_DIR / f"{account}.json"


def _list_accounts():
    """Return list of authenticated account names."""
    if not ACCOUNTS_DIR.exists():
        return []
    return sorted(p.stem for p in ACCOUNTS_DIR.glob("*.json"))


def _get_active_account():
    """Get the currently active account name."""
    cfg = get_section("calendar")
    # CLI-set active account takes priority
    if ACTIVE_ACCOUNT_PATH.exists():
        return ACTIVE_ACCOUNT_PATH.read_text().strip()
    # Config default
    default = cfg.get("default_account")
    if default:
        return default
    # Fall back to first account
    accounts = _list_accounts()
    return accounts[0] if accounts else None


def _set_active_account(account):
    """Set the active account."""
    ACTIVE_ACCOUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_ACCOUNT_PATH.write_text(account)


def _check_dep():
    try:
        import googleapiclient  # noqa: F401

        return True
    except ImportError:
        console.print(
            "[red]Google API client is not installed.[/red]\n"
            "Install with: [cyan]uv pip install murmur[calendar][/cyan]"
        )
        return False


def _get_credentials(account=None):
    """Load or create Google OAuth2 credentials for an account."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    if account is None:
        account = _get_active_account()
    if account is None:
        console.print(
            "[red]No accounts configured.[/red]\n"
            "Run [cyan]murmur calendar add <name>[/cyan] to add one."
        )
        raise SystemExit(1)

    token_file = _token_path(account)
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                console.print(
                    "[red]Google OAuth credentials not found.[/red]\n"
                    f"Download from Google Cloud Console and save to:\n"
                    f"  [cyan]{CREDENTIALS_PATH}[/cyan]\n\n"
                    "[dim]Steps:\n"
                    "1. Go to console.cloud.google.com \u2192 APIs & Services \u2192 Credentials\n"
                    "2. Create an OAuth 2.0 Client ID (type: Desktop app)\n"
                    "3. Download the JSON and save as google_credentials.json[/dim]"
                )
                raise SystemExit(1)

            console.print(
                f"[bold]Authenticating account:[/bold] [cyan]{account}[/cyan]\n"
                "A browser window will open for Google sign-in.\n"
                f"[yellow]Sign in with the account for '{account}'.[/yellow]\n"
            )
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return creds


def _build_service(account=None):
    """Build and return the Google Calendar API service."""
    from googleapiclient.discovery import build

    creds = _get_credentials(account)
    return build("calendar", "v3", credentials=creds)


def _fetch_events(time_min, time_max, max_results=20, account=None):
    """Fetch calendar events in a time range."""
    service = _build_service(account)
    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    return result.get("items", [])


def _fetch_all_accounts_events(time_min, time_max, max_results=20):
    """Fetch events from all authenticated accounts, merged and sorted."""
    accounts = _list_accounts()
    if not accounts:
        return []

    all_events = []
    for account in accounts:
        try:
            events = _fetch_events(time_min, time_max, max_results, account=account)
            for e in events:
                parsed = _parse_event(e)
                parsed["account"] = account
                all_events.append(parsed)
        except Exception as exc:
            console.print(f"[yellow]Warning: could not fetch from {account}: {exc}[/yellow]")

    all_events.sort(key=lambda e: e["start"] or datetime.min.replace(tzinfo=UTC))
    return all_events


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------


def _parse_event(event):
    """Parse a Google Calendar event into a clean dict."""
    start_raw = event["start"].get("dateTime", event["start"].get("date"))
    end_raw = event["end"].get("dateTime", event["end"].get("date"))

    try:
        start_dt = datetime.fromisoformat(start_raw)
        end_dt = datetime.fromisoformat(end_raw)
        duration_mins = int((end_dt - start_dt).total_seconds() / 60)
    except ValueError, TypeError:
        start_dt = None
        end_dt = None
        duration_mins = None

    attendees = []
    for a in event.get("attendees", []):
        name = a.get("displayName") or a.get("email", "")
        status = a.get("responseStatus", "needsAction")
        if a.get("self"):
            continue
        attendees.append({"name": name, "status": status})

    meet_link = None
    conf = event.get("conferenceData", {})
    for entry in conf.get("entryPoints", []):
        if entry.get("entryPointType") == "video":
            meet_link = entry.get("uri")
            break
    if not meet_link:
        meet_link = event.get("hangoutLink")

    return {
        "id": event.get("id"),
        "title": event.get("summary", "(no title)"),
        "start": start_dt,
        "end": end_dt,
        "duration_mins": duration_mins,
        "attendees": attendees,
        "meet_link": meet_link,
        "location": event.get("location"),
        "description": event.get("description"),
        "organizer": event.get("organizer", {}).get("displayName")
        or event.get("organizer", {}).get("email"),
        "account": None,
    }


# ---------------------------------------------------------------------------
# Public API (used by summarizer and other plugins)
# ---------------------------------------------------------------------------


def get_today_events(account=None, all_accounts=False):
    """Fetch all events for today."""
    now = datetime.now(tz=UTC)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    if all_accounts:
        return _fetch_all_accounts_events(start_of_day, end_of_day)
    return [_parse_event(e) for e in _fetch_events(start_of_day, end_of_day, account=account)]


def get_current_event(account=None, all_accounts=False):
    """Get the event happening right now, if any."""
    now = datetime.now(tz=UTC)
    events = get_today_events(account=account, all_accounts=all_accounts)
    for event in events:
        if event["start"] and event["end"] and event["start"] <= now <= event["end"]:
            return event
    return None


def get_next_event(account=None):
    """Get the next upcoming event."""
    now = datetime.now(tz=UTC)
    end_of_day = now.replace(hour=23, minute=59, second=59)
    raw = _fetch_events(now, end_of_day, max_results=1, account=account)
    events = [_parse_event(e) for e in raw]
    return events[0] if events else None


def match_recording_to_event(recording_time):
    """Find the calendar event that overlaps with a recording timestamp.

    Searches all accounts for the best match.
    """
    window_start = recording_time - timedelta(minutes=15)
    window_end = recording_time + timedelta(minutes=15)
    events = _fetch_all_accounts_events(window_start, window_end)

    for event in events:
        if (
            event["start"]
            and event["end"]
            and event["start"] - timedelta(minutes=10) <= recording_time <= event["end"]
        ):
            return event
    return None


def event_to_context(event):
    """Convert a calendar event into context string for the summarizer."""
    parts = [f"Meeting: {event['title']}"]
    if event.get("account"):
        parts.append(f"Calendar account: {event['account']}")
    if event["organizer"]:
        parts.append(f"Organizer: {event['organizer']}")
    if event["attendees"]:
        names = [a["name"] for a in event["attendees"]]
        parts.append(f"Attendees: {', '.join(names)}")
    if event["duration_mins"]:
        parts.append(f"Scheduled duration: {event['duration_mins']} minutes")
    if event["description"]:
        desc = event["description"][:500]
        parts.append(f"Meeting description: {desc}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _render_events_table(events, title="Meetings", show_account=False):
    """Render a Rich table of calendar events."""
    table = Table(title=title, expand=True)
    table.add_column("Time", style="cyan", width=14)
    table.add_column("Duration", style="yellow", width=8)
    table.add_column("Meeting", style="white")
    table.add_column("Attendees", style="dim")
    if show_account:
        table.add_column("Account", style="magenta", width=16)
    table.add_column("Link", style="blue", width=5)

    now = datetime.now(tz=UTC)

    for event in events:
        if event["start"]:
            time_str = event["start"].strftime("%H:%M")
            if event["start"] <= now <= (event["end"] or now):
                time_str = f"[bold red]\u25cf {time_str}[/bold red]"
        else:
            time_str = "all day"

        dur = f"{event['duration_mins']}m" if event["duration_mins"] else "\u2014"
        attendee_names = ", ".join(a["name"].split("@")[0] for a in event["attendees"][:4])
        if len(event["attendees"]) > 4:
            attendee_names += f" +{len(event['attendees']) - 4}"
        link = "Meet" if event["meet_link"] else "\u2014"

        row = [time_str, dur, event["title"], attendee_names]
        if show_account:
            row.append(event.get("account") or "\u2014")
        row.append(link)
        table.add_row(*row)

    return table


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(cli):
    """Register the calendar command group."""

    @cli.group(invoke_without_command=True)
    @click.option(
        "-a",
        "--account",
        default=None,
        help="Account to use (default: active account).",
    )
    @click.option(
        "--all",
        "all_accounts",
        is_flag=True,
        default=False,
        help="Show events from all accounts.",
    )
    @click.pass_context
    def calendar(ctx, account, all_accounts):
        """Google Calendar integration with multi-account support."""
        ctx.ensure_object(dict)
        ctx.obj["account"] = account
        ctx.obj["all_accounts"] = all_accounts
        if ctx.invoked_subcommand is not None:
            return
        ctx.invoke(today)

    @calendar.command()
    @click.pass_context
    def today(ctx):
        """List today's meetings."""
        if not _check_dep():
            raise SystemExit(1)

        account = ctx.obj.get("account")
        all_accts = ctx.obj.get("all_accounts", False)

        if all_accts:
            events = get_today_events(all_accounts=True)
            show_account = True
        else:
            events = get_today_events(account=account)
            show_account = False

        if not events:
            console.print("[dim]No meetings today.[/dim]")
            return

        active = account or _get_active_account()
        title = "Today's Meetings (all accounts)" if all_accts else f"Today's Meetings ({active})"
        console.print(_render_events_table(events, title=title, show_account=show_account))

    @calendar.command()
    @click.pass_context
    def next(ctx):
        """Show the next upcoming meeting."""
        if not _check_dep():
            raise SystemExit(1)

        account = ctx.obj.get("account")
        event = get_next_event(account=account)
        if not event:
            console.print("[dim]No more meetings today.[/dim]")
            return

        console.print(f"[bold]{event['title']}[/bold]")
        if event["start"]:
            delta = event["start"] - datetime.now(tz=UTC)
            mins = int(delta.total_seconds() / 60)
            if mins > 0:
                console.print(f"  Starts in [cyan]{mins} minutes[/cyan]")
            else:
                console.print("  [bold red]Happening now[/bold red]")
        if event["attendees"]:
            names = ", ".join(a["name"] for a in event["attendees"])
            console.print(f"  Attendees: {names}")
        if event["meet_link"]:
            console.print(f"  Link: [blue]{event['meet_link']}[/blue]")

    @calendar.command()
    @click.pass_context
    def current(ctx):
        """Show the meeting happening right now."""
        if not _check_dep():
            raise SystemExit(1)

        account = ctx.obj.get("account")
        all_accts = ctx.obj.get("all_accounts", False)
        event = get_current_event(account=account, all_accounts=all_accts)
        if not event:
            console.print("[dim]No meeting happening right now.[/dim]")
            return

        console.print(f"[bold red]\u25cf[/bold red] [bold]{event['title']}[/bold]")
        if event.get("account"):
            console.print(f"  Account: [magenta]{event['account']}[/magenta]")
        if event["start"] and event["end"]:
            elapsed = datetime.now(tz=UTC) - event["start"]
            remaining = event["end"] - datetime.now(tz=UTC)
            console.print(
                f"  Elapsed: [yellow]{int(elapsed.total_seconds() / 60)}m[/yellow]  "
                f"Remaining: [cyan]{int(remaining.total_seconds() / 60)}m[/cyan]"
            )
        if event["attendees"]:
            names = ", ".join(a["name"] for a in event["attendees"])
            console.print(f"  Attendees: {names}")
        if event["meet_link"]:
            console.print(f"  Link: [blue]{event['meet_link']}[/blue]")

    # --- Account management commands ---

    @calendar.command()
    @click.argument("name")
    def add(name):
        """Add and authenticate a Google Calendar account.

        NAME is a label for this account (e.g., 'personal', 'work', 'client-a').
        """
        if not _check_dep():
            raise SystemExit(1)

        _get_credentials(account=name)
        # Set as active if it's the first account
        if len(_list_accounts()) == 1:
            _set_active_account(name)
        console.print(f"[bold green]Account '{name}' added and authenticated.[/bold green]")

    @calendar.command()
    def accounts():
        """List all authenticated accounts."""
        accts = _list_accounts()
        if not accts:
            console.print("[dim]No accounts configured.[/dim]")
            console.print("Run [cyan]murmur calendar add <name>[/cyan] to add one.")
            return

        active = _get_active_account()
        table = Table(title="Calendar Accounts")
        table.add_column("Account", style="cyan")
        table.add_column("Active", style="green", width=8)
        table.add_column("Token", style="dim")

        for acct in accts:
            is_active = "\u2713" if acct == active else ""
            token_file = _token_path(acct)
            token_status = "valid" if token_file.exists() else "missing"
            table.add_row(acct, is_active, token_status)

        console.print(table)

    @calendar.command(name="use")
    @click.argument("name")
    def use_account(name):
        """Switch the active calendar account."""
        accts = _list_accounts()
        if name not in accts:
            console.print(f"[red]Account '{name}' not found.[/red]")
            console.print(f"Available: {', '.join(accts)}")
            return
        _set_active_account(name)
        console.print(f"[bold green]Switched to account '{name}'.[/bold green]")

    @calendar.command()
    @click.argument("name")
    def remove(name):
        """Remove an authenticated account."""
        token_file = _token_path(name)
        if not token_file.exists():
            console.print(f"[red]Account '{name}' not found.[/red]")
            return
        token_file.unlink()
        # If removing active account, clear it
        if _get_active_account() == name:
            ACTIVE_ACCOUNT_PATH.unlink(missing_ok=True)
        console.print(f"[yellow]Account '{name}' removed.[/yellow]")
