"""
Tests for live_scraper module.
"""

import unittest
from src.live_scraper import AMDDocsScraper


class TestLiveScraper(unittest.TestCase):

    def test_scraper_init(self):
        scraper = AMDDocsScraper()
        self.assertIsNotNone(scraper.session)
        self.assertEqual(len(scraper.URLS), 4)

    def test_validate_urls(self):
        scraper = AMDDocsScraper()
        test_urls = [
            'https://rocm.docs.amd.com/projects/install-on-linux/en/latest/',
        ]
        res = scraper.validate_urls(test_urls)
        self.assertIn(test_urls[0], res)
        # It should be reachable or at least get a response status code
        self.assertTrue(res[test_urls[0]].get('reachable', False) or 'status_code' in res[test_urls[0]])


if __name__ == '__main__':
    unittest.main()
