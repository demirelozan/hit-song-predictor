from song_definition import Song


class CategoricalSongData(Song):
    categorical_features = ['title', 'date', 'simple_title', 'artist', 'main_artist', 'spotify_link', 'spotify_id',
                            'video_link', 'genre', 'analysis_url', 'lyrics']

    def __init__(self, title, artist, peak_pos, lyrics, date, main_artist=None, change=None, spotify_link=None,
                 spotify_id=None, video_link=None, genre=None, broad_genre=None, analysis_url=None,
                 additional_spotify_attributes=None):
        """
        Initialize the CategoricalSongData class with categorical attributes.

        :param title: str - Title of the song.
        :param artist: str - Artist of the song.
        :param peak_pos: int - Peak position on the Billboard chart.
        :param lyrics: str - Lyrics of the song.
        :param date: str - Date of the song's chart appearance.
        :param main_artist: str - Main artist of the song.
        :param change: str - Change in chart position.
        :param spotify_link: str - Link to the song on Spotify.
        :param spotify_id: str - Spotify ID of the song.
        :param video_link: str - Link to the song's video.
        :param genre: str - Genre of the song.
        :param broad_genre: str - Broader genre categorization.
        :param analysis_url: str - URL for detailed song analysis.
        :param additional_spotify_attributes: dict - Other Spotify attributes.
        """
        super().__init__(title, artist, peak_pos)
        self.lyrics = lyrics
        self.date = date
        self.main_artist = main_artist
        self.change = change
        self.spotify_link = spotify_link
        self.spotify_id = spotify_id
        self.video_link = video_link
        self.genre = genre
        self.broad_genre = broad_genre
        self.analysis_url = analysis_url
        self.additional_spotify_attributes = additional_spotify_attributes or {}

    def process_lyrics(self):
        """
        Process the lyrics for NLP tasks.
        """
        # NLP processing logic goes here

    # Additional methods for handling and processing other categorical data can be added here
