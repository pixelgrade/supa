import concurrent.futures
import subprocess
from pathlib import Path

import pytest

from services import repo_evidence as repo


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init')
    git(origin, 'config', 'user.email', 'test@example.com')
    git(origin, 'config', 'user.name', 'Test')
    (origin / 'theme.php').write_text('<?php // real source\n')
    git(origin, 'add', 'theme.php')
    git(origin, 'commit', '-m', 'Initial')
    root = tmp_path / 'cache'
    root.mkdir()
    monkeypatch.setattr(repo, 'REPO_CACHE_DIR', root)
    monkeypatch.setattr(repo, 'repo_url', lambda tag, url=str(origin): url)
    return root, origin


@pytest.mark.parametrize('kind', ['empty', 'invalid-git', 'missing-source'])
def test_recovers_invalid_cache_without_discarding_it(cache, kind):
    root, origin = cache
    broken = root / 'field-journal'
    if kind == 'missing-source':
        subprocess.run(['git', 'clone', str(origin), str(broken)], check=True, capture_output=True)
        (broken / 'theme.php').unlink()
    else:
        broken.mkdir()
        if kind == 'invalid-git':
            (broken / '.git').mkdir()
    marker = broken / 'owner-marker'
    marker.write_text('preserve me')
    recovered = repo.ensure_repo_checkout('field-journal')
    assert (recovered / 'theme.php').read_text().startswith('<?php')
    assert marker.read_text() == 'preserve me'
    assert repo.ensure_repo_checkout('field-journal') == recovered


def test_failed_refresh_preserves_active_checkout(cache, monkeypatch):
    root, origin = cache
    active = repo.ensure_repo_checkout('field-journal')
    monkeypatch.setattr(repo, 'repo_url', lambda tag, url=str(root / 'absent-origin'): url)
    with pytest.raises(subprocess.CalledProcessError):
        repo.ensure_repo_checkout('field-journal', refresh=True)
    assert (active / 'theme.php').is_file()
    assert repo.ensure_repo_checkout('field-journal') == active


def test_refresh_keeps_old_readers_on_unchanged_generation(cache):
    root, origin = cache
    old = repo.ensure_repo_checkout('field-journal')
    (origin / 'theme.php').write_text('<?php // updated\n')
    git(origin, 'commit', '-am', 'Update')
    new = repo.ensure_repo_checkout('field-journal', refresh=True)
    assert new != old
    assert 'real source' in (old / 'theme.php').read_text()
    assert 'updated' in (new / 'theme.php').read_text()
    assert repo.ensure_repo_checkout('field-journal') == new


def test_concurrent_cold_calls_publish_one_generation(cache):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: repo.ensure_repo_checkout('field-journal'), range(4)))
    assert len(set(paths)) == 1
    assert all((path / 'theme.php').is_file() for path in paths)


def test_failed_rebuild_preserves_broken_cache(cache, monkeypatch):
    root, origin = cache
    broken = root / 'field-journal'
    broken.mkdir()
    (broken / 'owner-marker').write_text('preserve me')
    monkeypatch.setattr(repo, 'repo_url', lambda tag, url=str(root / 'absent-origin'): url)
    with pytest.raises(subprocess.CalledProcessError):
        repo.ensure_repo_checkout('field-journal')
    assert (broken / 'owner-marker').read_text() == 'preserve me'


def test_valid_legacy_checkout_reused_without_network(cache, monkeypatch):
    root, origin = cache
    legacy = root / 'field-journal'
    subprocess.run(['git', 'clone', str(origin), str(legacy)], check=True, capture_output=True)
    monkeypatch.setattr(repo, 'repo_url', lambda tag, url=str(root / 'absent-origin'): url)
    assert repo.ensure_repo_checkout('field-journal') == legacy


def test_concurrent_processes_publish_one_generation(cache):
    import sys
    root, origin = cache
    code = '''
import sys
from pathlib import Path
from services import repo_evidence as repo
repo.REPO_CACHE_DIR = Path(sys.argv[1])
repo.repo_url = lambda tag: sys.argv[2]
print(repo.ensure_repo_checkout('field-journal'))
'''
    children = [subprocess.Popen([sys.executable, '-c', code, str(root), str(origin)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for _ in range(4)]
    paths = []
    for child in children:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stderr
        paths.append(Path(stdout.strip()))
    assert len(set(paths)) == 1
    assert (paths[0] / 'theme.php').is_file()
