from data_model.song_definition import Song


class SongClassifier:
    def __init__(self, songs):
        self.songs = songs

    def classify_hits(self,  hit_threshold=10.0):
        """
        Classify songs as hits based on a threshold.

        :param hit_threshold: float - The chart position threshold to classify a song as a hit.
        """

        for song in self.songs:
            # Convert 'peak_pos' to float if it's a string
            if isinstance(song.peak_pos, str):
                song.peak_pos = float(song.peak_pos)
            song.is_hit = song.peak_pos <= hit_threshold

    def calculate_custom_score(self, weight_factors):
        """
        Calculate a custom score for each song based on provided weight factors.

        :param weight_factors: dict - A dictionary of weights for various song attributes.
        :return: None - The scores are stored or processed within the method.
        """
        for song in self.songs:
            # Example score calculation
            song.score = (weight_factors['peak_pos_weight'] * (1 - song.normalized_peak_pos) +
                          weight_factors['weeks_on_chart_weight'] * song.normalized_weeks_on_chart +
                          weight_factors['streaming_numbers_weight'] * song.normalized_streaming_numbers)
            # Add more factors as necessary

    def update_songs_from_data(self, data):
        """
        Update the list of songs based on the latest DataFrame.

        :param data: DataFrame - The updated dataset containing song information.
        """
        self.songs = [Song(row['title'], row['artist'], row['peak_pos'],
                           additional_attributes={'normalized_peak_pos': row.get('normalized_peak_pos', None),
                                                  'normalized_weeks_on_chart': row.get('normalized_weeks_on_chart',
                                                                                       None),
                                                  'normalized_streaming_numbers': row.get(
                                                      'normalized_streaming_numbers', None)})
                      for index, row in data.iterrows()]

    def execute_classification(self, data, hit_threshold=10, weight_factors=None, output_file_path = None):
        """
        Execute the classification and scoring of songs.

        :param output_file_path: The path where the output will be saved as an excel file
        :param data:
        :param hit_threshold: int - The threshold for classifying a song as a hit.
        :param weight_factors: dict - Weights for various song attributes for scoring.
        """
        self.classify_hits(hit_threshold)
        if weight_factors:
            self.calculate_custom_score(weight_factors)

        data['Is Hit'] = [int(song.is_hit) for song in self.songs]
        if weight_factors:
            data['Score'] = [song.score if hasattr(song, 'score') else 'N/A' for song in self.songs]

        # Save the updated DataFrame to an Excel file
        if output_file_path:
            data.to_excel(output_file_path, index=False)