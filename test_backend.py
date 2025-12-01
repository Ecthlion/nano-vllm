# test
from nanovllm.utils.backend import BackendAPI

if __name__ == "__main__":
    api = BackendAPI()
    api.load_data("./uploads/s2orc.csv")
    api.text_field = "abstract"
    # query = f'sentiment == "positive" and LLM("Given the above film review, answer whether it contains names. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    # query = f'LLM("Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n") == "positive"'
    # query = f'LLM("Given the above paper abstract, answer whether the proposed method outperform existing baselines. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    query = f'LLM("Given the above paper abstract, answer whether the paper is based on a new dataset. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    api.analyse(query)
    # 0.5 -> 0.88

    # api = BackendAPI()
    # api.load_data("./uploads/imdb.csv")
    # api.text_field = "review"
    # # query = f'sentiment == "positive" and LLM("Given the above film review, answer whether it contains names. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    # # query = f'LLM("Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n") == "positive"'
    # query = f'LLM("Given the above film review, answer whether it contains names. Respond ONLY with "yes" or "no", in all lower case.\n") == "yes"'
    # api.analyse(query)
