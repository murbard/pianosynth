import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from pianosynth.data import parse_html_for_links, download_iowa_piano_data, download_file

SAMPLE_HTML = """
<html>
<body>
<a href="sound files/MIS/Piano_Other/piano/Piano.pp.C4.aiff">C4 pp</a>
<a href="sound files/MIS/Piano_Other/piano/Piano.mf.A4.aiff">A4 mf</a>
</body>
</html>
"""

class TestDataDownload(unittest.TestCase):
    def test_parse_html(self):
        links = parse_html_for_links(SAMPLE_HTML)
        self.assertEqual(len(links), 2)
        # Check first match
        self.assertEqual(links[0][1], "pp") # dynamic
        self.assertEqual(links[0][2], "C4") # note
        # Check second match
        self.assertEqual(links[1][1], "mf")
        self.assertEqual(links[1][2], "A4")

    @patch('requests.get')
    def test_download_file(self, mock_get):
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response
        
        with patch('builtins.open', mock_open()) as mock_file:
            download_file("http://example.com/file", "dummy_path")
            
            mock_file.assert_called_with("dummy_path", "wb")
            handle = mock_file()
            handle.write.assert_any_call(b'chunk1')
            handle.write.assert_any_call(b'chunk2')

    @patch('pianosynth.data.download_file')
    @patch('pianosynth.data.requests.get') 
    @patch('pianosynth.data.open', new_callable=mock_open, read_data=SAMPLE_HTML)
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_download_logic(self, mock_mkdir, mock_exists, mock_file_open, mock_requests_get, mock_download_file):
        # Scenario: HTML exists, but files do not
        # mock_exists side_effect: 
        # 1. HTML path exists? -> False (so we trigger download index logic)
        # 2. Save path exists? -> False (so we trigger sample download)
        
        # Simulating that HTML file does NOT exist initially
        mock_exists.side_effect = [False, False, False] 
        
        mock_requests_get.return_value.text = SAMPLE_HTML
        
        download_iowa_piano_data("dummy_dir")
        
        # Verify index download
        mock_requests_get.assert_called_with("http://theremin.music.uiowa.edu/MISpiano.html")
        
        # Verify parsing and download calls
        # We expect 2 downloads based on SAMPLE_HTML
        self.assertEqual(mock_download_file.call_count, 2)
