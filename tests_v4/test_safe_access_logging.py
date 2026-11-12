"""Access log privacy without changing legacy request authentication transport."""
import logging

import pytest

from arborscan_v4.safe_logging import AccessPathFilter, install_access_log_filter


def record(args):
    item = logging.LogRecord('uvicorn.access', logging.INFO, __file__, 1,
                             '%s - "%s %s HTTP/%s" %d', (), None)
    item.args = args
    return item


@pytest.mark.parametrize('path,expected', [
    ('/analyses/123?token=synthetic-private-token&x=2', '/analyses/123'),
    ('/v4/health', '/v4/health'),
    ('/x?%74oken=private%3Fdata', '/x'),
])
def test_standard_access_record_removes_entire_query(path, expected):
    args = ('127.0.0.1:123', 'GET', path, '1.1', 200)
    item = record(args)
    assert AccessPathFilter().filter(item) is True
    assert item.args == ('127.0.0.1:123', 'GET', expected, '1.1', 200)
    assert '?' not in item.getMessage()
    assert args[2] == path  # Caller/request state stays untouched.


@pytest.mark.parametrize('args', [(), ('one',), {'path': '?token=private'},
                                  ('ip', 'GET', object(), '1.1', 200)])
def test_unexpected_record_shape_is_dropped_without_formatting_secret(args):
    assert AccessPathFilter().filter(record(args)) is False


def test_install_is_idempotent_and_keeps_other_logger_configuration():
    logger = logging.getLogger('uvicorn.access')
    previous = list(logger.filters)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    try:
        logger.filters = []
        install_access_log_filter()
        install_access_log_filter()
        assert len(logger.filters) == 1
        assert isinstance(logger.filters[0], AccessPathFilter)
        assert (list(logger.handlers), logger.level, logger.propagate) == (handlers, level, propagate)
    finally:
        logger.filters = previous
