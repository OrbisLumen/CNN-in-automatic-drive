"""Offline downloads, cache reuse, cleanup, and progress reporting."""

from pathlib import Path

import pytest

from mydezero import utils


@pytest.mark.parametrize('total,expected', [(-1, '128 bytes downloaded'), (0, '128 bytes downloaded'), (100, '100.00%')])
def test_progress_unknown_zero_and_final_block(total, expected, capsys):
    utils.show_progress(2, 64, total)
    text = capsys.readouterr().out
    assert text.startswith('\r') and expected in text
    if total > 0:
        assert '[' + '#' * 30 + ']' in text


def test_progress_midpoint(capsys):
    utils.show_progress(1, 50, 100)
    assert capsys.readouterr().out == '\r[' + '#' * 15 + '.' * 15 + '] 50.00%'


def test_download_local_file_creates_nested_cache_and_reuses_it(tmp_path, monkeypatch):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'example payload')
    cache = tmp_path / 'cache' / 'nested'
    result = utils.get_file(source.as_uri(), cache_dir=cache)
    assert Path(result).read_bytes() == b'example payload'
    assert Path(result).is_absolute()
    monkeypatch.setattr(utils.urllib.request, 'urlretrieve', lambda *args: pytest.fail('cache was not reused'))
    assert utils.get_file(source.as_uri(), cache_dir=cache) == result


def test_default_cache_and_url_query_filename(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    def retrieve(url, path, callback):
        Path(path).write_bytes(b'content')
        callback(1, 7, 7)

    monkeypatch.setattr(utils.urllib.request, 'urlretrieve', retrieve)
    result = utils.get_file('https://example.com/data.gz?token=abc#fragment')
    assert Path(result) == tmp_path / '.mydezero' / 'data.gz'
    assert Path(result).read_bytes() == b'content'


def test_explicit_filename_and_expanded_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))
    source = tmp_path / 'source'
    source.write_bytes(b'payload')
    result = utils.get_file(source.as_uri(), file_name='renamed.bin', cache_dir='~/cache')
    assert Path(result) == tmp_path / 'cache' / 'renamed.bin'
    assert Path(result).read_bytes() == b'payload'


@pytest.mark.parametrize('error', [OSError('network failed'), KeyboardInterrupt()])
@pytest.mark.parametrize('partial', [True, False])
def test_download_failure_cleans_partial_file_and_propagates(tmp_path, monkeypatch, error, partial):
    def retrieve(url, path, callback):
        if partial:
            Path(path).write_bytes(b'incomplete')
        raise error

    monkeypatch.setattr(utils.urllib.request, 'urlretrieve', retrieve)
    with pytest.raises(type(error)) as caught:
        utils.get_file('https://example.com/data.gz', cache_dir=tmp_path)
    assert caught.value is error
    assert not (tmp_path / 'data.gz').exists()
