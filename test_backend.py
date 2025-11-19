# test
from nanovllm.utils.backend import BackendAPI

if __name__ == "__main__":
    api = BackendAPI()
    api.load_data("./data/imdb.csv")
    api.build_index(0.9, "review")
