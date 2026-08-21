"""
Description: Unit tests for backend.utils.link_discovery.
    Each test checks one specific behavior/edge case in isolation, so if
    this function ever breaks, we know exactly which case failed.
"""
from sympy import true

from backend.utils import link_discovery

link_discoverer=link_discovery.LinkDiscoverer("http://iitpkd.ac.in",25)
def test_is_worth_crawling_non_iitpkd_domain():
    """
    Description:  tests whether _is_worth_crawling opens the website belongs to
    iitpkd.ac.in domain if not it skips
    """
    url = "http://google.com"
    assert link_discoverer._is_worth_crawling(url)==False
    assert link_discoverer._is_worth_crawling("http://sac.iitpkd.ac.in")==True
    assert link_discoverer._is_worth_crawling("http://iitpk.ac.in")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.com")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.in")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.ac")==False

def test_is_worth_crawling_non_webpage_domain():
    """
    Description: tests if _is_worth_crawling opens a link that points to a webpage and
    skips a url that points to a pdf, png, jpeg, docx
    """
    assert link_discoverer._is_worth_crawling("http://iitpkd.pdf")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.png")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.jpeg")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.jpg")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.zip")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.docs")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.ac.in/docs")==True


def test_is_worth_crawling_in_discovered_urls():
    """
    Description: tests whether _is_worth_crawling opens the website whose url
    is not in the discovered urls list
    """
    link_discoverer.discovered_urls = ["http://iitpkd.ac.in", "http://sac.iitpkd.ac.in"]
    assert link_discoverer._is_worth_crawling("http://iitpkd.ac.in")==False
    assert link_discoverer._is_worth_crawling("http://iitpkd.ac.in/teaching")==True


