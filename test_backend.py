# test
from nanovllm.utils.backend import BackendAPI

if __name__ == "__main__":
    api = BackendAPI()
    api.load_data("./data/imdb.csv")
    # api.build_index(0.9, "review")
    # query = f'sentiment == "positive" and LLM("Given the above film review, answer whether it contains names. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    # query = f'LLM("Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n") == "positive"'
    query = f'LLM("Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    api.analyse(query)
