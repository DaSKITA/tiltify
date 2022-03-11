from tiltify.data_loader import DataLoader


class TestDataLoader:

    def test_get_json_data(self):
        # setting up
        test_folder = "test_data/policies"
        data_loader = DataLoader()

        # execute
        loaded_files = data_loader.get_json_data(test_folder)

        # assert
        assert len(loaded_files) == 1
        assert isinstance(loaded_files[0], dict)
        assert loaded_files[0]["document"]["document_name"] is not None
