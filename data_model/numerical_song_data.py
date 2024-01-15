class NumericalSongData:
    # List of numerical feature names
    numerical_features = ['year', 'peak_pos', 'last_pos', 'weeks', 'rank',
                          'energy', 'liveness', 'tempo', 'speechiness',
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
