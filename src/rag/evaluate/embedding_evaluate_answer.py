from sklearn.metrics.pairwise import cosine_similarity

def semantic_match(embed_model, answer, gt):

    vec1 = embed_model.embed_query(answer)
    vec2 = embed_model.embed_query(gt)

    score = cosine_similarity([vec1], [vec2])[0][0]

    return score