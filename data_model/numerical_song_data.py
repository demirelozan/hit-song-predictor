from song_definition import Song

class NumericalSongData(Song):
    def __init__(self, title, artist, peak_pos, year, last_pos, weeks, rank, spotify_attributes):
        """
        Initialize the NumericalSongData class with numerical attributes.

        :param title: str - Title of the song
        :param artist: str - Artist of the song
        :param peak_pos: int - Peak position on the Billboard chart
        :param year: int - Year the song was on the Billboard chart
        :param last_pos: int - Last position of the song on the Billboard chart
        :param weeks: int - Number of weeks the song was on the Billboard chart
        :param rank: int - Rank of the song
        :param spotify_attributes: dict - Spotify related attributes
        """
        super().__init__(title, artist, peak_pos)
        self.year = year
        self.last_pos = last_pos
        self.weeks = weeks
        self.rank = rank

        # Process and store Spotify attributes
        self.process_spotify_attributes(spotify_attributes)

    def process_spotify_attributes(self, attributes):
        """
        Process and store Spotify attributes.

        :param attributes: dict - Spotify related attributes
        """
        self.acousticness = attributes.get('acousticness')
        self.danceability = attributes.get('danceability')
        self.duration_ms = attributes.get('duration_ms')
        self.instrumentalness = attributes.get('instrumentalness')
        self.key = attributes.get('key')
        self.loudness = attributes.get('loudness')
        self.mode = attributes.get('mode')
        self.tempo = attributes.get('tempo')
        self.time_signature = attributes.get('time_signature')
        self.valence = attributes.get('valence')

        # Additional processing can be added here as needed