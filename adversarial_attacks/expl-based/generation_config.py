GENERATION_CONFIG = {
    'greedy_search': {
        "max_length": 50,
        "min_length": 0,
        "early_stopping": False,
        "do_sample": False,
        "num_beams": 1,
        "penalty_alpha": None,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "no_repeat_ngram_size": 0
    },
    'beam_search': {
        "max_length": 50,
        "min_length": 0,
        "early_stopping": True,
        "do_sample": False,
        "num_beams": 4,
        "penalty_alpha": None,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "no_repeat_ngram_size": 0
    },
    'top-k_sampling': {
        "max_length": 50,
        "min_length": 0,
        "early_stopping": True,
        "do_sample": True,
        "num_beams": 1,
        "penalty_alpha": None,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "no_repeat_ngram_size": 0
    },
    'top-p_sampling': {
        "max_length": 50,
        "min_length": 0,
        "early_stopping": True,
        "do_sample": True,
        "num_beams": 1,
        "penalty_alpha": None,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.9,
        "no_repeat_ngram_size": 0
    }
}