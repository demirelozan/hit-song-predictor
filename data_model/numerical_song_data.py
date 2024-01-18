class NumericalSongData:
    # List of numerical feature names
    numerical_features = ['peak_pos', 'last_pos', 'weeks', 'change', 'rank',
                          'energy', 'liveness', 'tempo', 'speechiness',
                          'acousticness', 'instrumentalness', 'time_signature',
                          'danceability', 'key', 'duration_ms', 'loudness',
                          'valence', 'mode', 'is_hit']
    spotify_numerical_features = ['energy', 'liveness', 'tempo', 'speechiness',
                                  'acousticness', 'instrumentalness', 'time_signature',
                                  'danceability', 'key', 'duration_ms', 'loudness',
                                  'valence', 'mode']

    def __init__(self):
        pass

    @staticmethod
    def get_numerical_features():
        """
        Static method to get the list of numerical features.
        :return: List of numerical feature names.
        """
        return NumericalSongData.numerical_features

    @staticmethod
    def get_spotify_numerical_features():
        return NumericalSongData.spotify_numerical_features
