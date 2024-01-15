class CategoricalSongData:
    # List of categorical feature names
    categorical_features = ['title', 'artist', 'lyrics', 'genre', 'main_artist',
                            'spotify_link', 'spotify_id', 'video_link',
                            'simple_title', 'analysis_url']

    def __init__(self):
        pass

    @staticmethod
    def get_categorical_features():
        """
        Static method to get the list of categorical features.
        :return: List of categorical feature names.
        """
        return CategoricalSongData.categorical_features

