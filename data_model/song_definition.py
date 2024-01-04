class Song:
    def __init__(self, title, artist, peak_pos, additional_attributes=None):
        """
        Initialize the Song class with basic attributes and additional attributes.

        :param title: str - Title of the song
        :param artist: str - Artist of the song
        :param peak_pos: int - Peak position on the Billboard chart
        :param additional_attributes: dict - A dictionary of additional attributes like Spotify features
        """
        self.title = title
        self.artist = artist
        self.peak_pos = peak_pos
        self.additional_attributes = additional_attributes or {}

    def is_hit(self, hit_threshold=10):
        """
        Determine if the song is a hit based on its peak position.

        :param hit_threshold: int - The threshold for a song to be considered a hit (default is top 10)
        :return: bool - True if the song is a hit, False otherwise
        """
        return self.peak_pos <= hit_threshold

    # Add any additional methods as needed, like for processing lyrics, extracting features, etc.
