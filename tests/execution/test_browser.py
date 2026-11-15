"""Test the browser a site is driven through.

Nothing here reaches a bookmaker. The pages are served over loopback, which is the only kind of site a test may open.
"""

import asyncio
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sportsbet.core import BuildError, ExecutionError, VenueBlockedError
from sportsbet.execution import BaseVenue, BrowserSession, build_venue

SLIP = """<!doctype html><html><body>
<form aria-label="Bet slip">
  <label for="s">Stake</label><input id="s" value="10.00">
  <button id="ready" type="button">Place bet</button>
  <button id="notready" type="button" disabled>Confirm bet</button>
  <button id="hidden" type="button" style="display:none">Hidden bet</button>
</form>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """Serve the pages a test arranged, and nothing else."""

    def __init__(self, pages, *args, **rest):
        """Keep the pages the test arranged."""
        self.pages = pages
        super().__init__(*args, **rest)

    def do_GET(self):
        """Answer with the page arranged for this path."""
        body, status = self.pages.get(self.path, ('not found', 404))
        self.send_response(status)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        """Say nothing, since a test is not a web server."""


@pytest.fixture
def served():
    """Serve pages over loopback, which is the only kind of site a test may open."""
    pages = {}
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(_Handler, pages))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address

    def serve(body, path='/slip', status=200):
        pages[path] = (body, status)
        return f'http://{host}:{port}{path}'

    yield serve
    server.shutdown()


@pytest.fixture
def run():
    """Run coroutines in one event loop for the whole test.

    A browser is opened once and driven across calls, and playwright binds what it opens to the loop that opened it, so
    a loop per call leaves the second call talking to a loop that is gone.
    """
    loop = asyncio.new_event_loop()
    yield loop.run_until_complete
    loop.close()


@pytest.fixture
def session(run):
    """Return a session with no pacing, since a test is not being polite to anybody."""
    built = BrowserSession(key='loopback', url='http://127.0.0.1', min_interval=0.0, timeout=1200.0)
    yield built
    run(built.stop())


def test_the_session_is_not_a_venue():
    """The site path promises nothing it cannot keep.

    Placing at a site needs the site's own knowledge, so there is no `place` here and the agent places. Once-only and
    the ceilings hold where the library does the placing, which is a venue with an API. A caller must not be able to
    reach a guarantee that is not there, so it is not there to reach.
    """
    built = BrowserSession(key='example', url='https://example.invalid')
    assert not isinstance(built, BaseVenue)
    assert not issubclass(BrowserSession, BaseVenue)
    for absent in ('place', 'read_status', 'cancel', 'read_balance', 'list_markets'):
        assert not hasattr(built, absent), f'`{absent}` promises something this path cannot keep'


def test_a_session_is_named_the_way_a_model_is_named(tmp_path):
    """A session of your own is reachable from the surfaces, as a model of your own is.

    It is not a venue on purpose, so the check that a venue is a venue has to know that a browser session is the other
    thing a venue reference can name. Without this the whole site path is unreachable from the command line and the
    tools.
    """
    written = tmp_path / 'venue.py'
    written.write_text(
        "from sportsbet.execution import BrowserSession\n"
        "VENUE = BrowserSession(key='example', url='https://example.invalid')\n",
    )
    assert isinstance(build_venue(f'{written}:VENUE'), BrowserSession)


def test_something_that_is_neither_is_refused(tmp_path):
    """A reference that names something that is not a venue and not a session says so."""
    written = tmp_path / 'venue.py'
    written.write_text('VENUE = 42\n')
    with pytest.raises(BuildError, match='not a venue and is not a browser session'):
        build_venue(f'{written}:VENUE')


def test_the_notes_are_kept_exactly_and_never_read():
    """The site knowledge is the user's, stored as written and handed back as written."""
    notes = 'Bet history is under My Account. The slip opens on the right; confirm is two steps.'
    built = BrowserSession(key='example', url='https://example.invalid', notes=notes)
    assert built.notes == notes


def test_the_constructor_keeps_what_it_was_given():
    """Parameters are stored unmodified, as every estimator in this library stores them."""
    built = BrowserSession(key='example', url='https://example.invalid/x', notes='a', min_interval=2.5)
    assert (built.key, built.url, built.notes, built.min_interval) == ('example', 'https://example.invalid/x', 'a', 2.5)


def test_the_page_comes_back_with_a_ref_for_everything_actionable(session, served, run):
    """A page is read in a form an agent can reason about and act on."""
    snapshot = run(session.navigate(served(SLIP)))
    assert 'form "Bet slip"' in snapshot.yaml
    assert 'textbox "Stake"' in snapshot.yaml
    assert '[ref=' in snapshot.yaml


def test_a_disabled_confirm_is_not_clicked(session, served, run):
    """A confirm the site has disabled is not clicked, and this says so.

    This is why the browser is driven rather than the page injected with script. A false success on a confirm button is
    a receipt that lies about money.
    """
    snapshot = run(session.navigate(served(SLIP)))
    ref = _ref_of(snapshot.yaml, 'Confirm bet')
    assert '[disabled]' in snapshot.yaml
    with pytest.raises(Exception, match=r'Timeout|disabled'):
        run(session.click(ref))


def test_a_hidden_control_is_not_clicked(session, served, run):
    """A control the site is not showing is not clicked."""
    run(session.navigate(served(SLIP)))
    with pytest.raises(Exception, match=r'Timeout|hidden|not visible'):
        run(session.click('e99'))


def test_an_enabled_control_is_clicked(session, served, run):
    """A control the site is showing is clicked, so the refusal above is about the control and not about clicking."""
    snapshot = run(session.navigate(served(SLIP)))
    ref = _ref_of(snapshot.yaml, 'Place bet')
    after = run(session.click(ref))
    assert 'Bet slip' in after.yaml


def test_typing_shows_up_in_the_snapshot_it_returns(session, served, run):
    """An action returns the page it produced, so the agent sees what it did without asking again."""
    snapshot = run(session.navigate(served(SLIP)))
    ref = _ref_of(snapshot.yaml, 'Stake')
    after = run(session.type(ref, '25.00'))
    assert '25.00' in after.yaml


def test_a_blocked_site_is_reported_and_nothing_further_is_tried(session, served, run):
    """A site that blocks automation is reported, and nothing here gets around it."""
    with pytest.raises(VenueBlockedError, match='refused automated access'):
        run(session.navigate(served(SLIP, path='/blocked', status=403)))


def test_page_is_the_page_the_session_drives(session, served, run):
    """Test the session hands a browser venue the playwright page it is driving, at the page it navigated to."""
    url = served(SLIP)
    run(session.navigate(url))
    assert session.page_.url == url


def test_page_before_the_browser_is_open_says_so():
    """Test asking for the page before the browser opens says what is missing."""
    built = BrowserSession(key='example', url='https://example.invalid')
    with pytest.raises(ExecutionError, match='not open'):
        _ = built.page_


def test_acting_before_the_browser_is_open_says_so(run):
    """A call before the browser opens says what is missing."""
    built = BrowserSession(key='example', url='https://example.invalid')
    with pytest.raises(ExecutionError, match='not open'):
        run(built.click('e1'))


def _ref_of(yaml, name):
    """Return the ref of the element a snapshot shows under a name."""
    for line in yaml.splitlines():
        if f'"{name}"' in line and '[ref=' in line:
            return line.split('[ref=')[1].split(']')[0]
    msg = f'no ref for {name} in\n{yaml}'
    raise AssertionError(msg)
