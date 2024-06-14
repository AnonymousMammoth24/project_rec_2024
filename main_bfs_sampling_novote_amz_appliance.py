from dataset.avazu import AvazuDataset
from dataset.criteo import CriteoDataset
from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from dataset.amazon_beauty import AmazonDataset
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import time
import openai
from sklearn.metrics import roc_auc_score
import collections
import pickle
import re
import itertools
from sklearn.metrics.pairwise import cosine_similarity


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def decoder_for_gpt3(model_name, temperature, input, max_length):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)
    time.sleep(1)

    # https://beta.openai.com/account/api-keys
    openai.api_key = ''

    # Specify engine ...
    # Instruct GPT3
    if model_name == "gpt3":
        engine = "text-ada-001"
    elif model_name == "gpt3.5":
        engine = "gpt-4"
    elif model_name == "gpt3-medium":
        engine = "text-babbage-001"
    elif model_name == "gpt3-large":
        engine = "text-curie-001"
    elif model_name == "gpt3-xl":
        engine = "text-davinci-002"
    elif model_name == "text-davinci-001":
        engine = "text-davinci-001"
    elif model_name == "code-davinci-002":
        engine = "code-davinci-002"
    else:
        raise ValueError("model is not properly defined ...")

    if engine == "code-davinci-002":
        response = openai.Completion.create(
            engine=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        return response["choices"][0]["text"]
    elif engine == "gpt-4":
        messages = [{"role": "system", "content": 'You are a helpful recommender'}, {"role": "user", "content": input}]
        response = openai.ChatCompletion.create(
            model=engine,
            messages=messages,
            max_tokens=max_length,
            # temperature=temperature,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=None
        )
        return response["choices"][0]["message"]["content"]
    else:
        response = openai.Completion.create(
            engine=engine,  ##can be replaced by model=
            prompt=input,
            max_tokens=max_length,
            # temperature=temperature,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=None
        )
        return response["choices"][0]["text"]


def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt(prompt, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    time.sleep(1)
    openai.api_key = ''
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)


def chatgpt(messages, model="gpt-4", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = None
        try_time = 0
        while res is None:
            try_time += 1
            print('try times:', try_time)
            try:
                res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                       n=cnt, stop=stop)
            except:
                pass
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
    return outputs


class Decoder():
    def __init__(self):
        # print_now()
        pass

    def decode(self, model_name, temperature, input, max_length):
        response = decoder_for_gpt3(model_name, temperature, input, max_length)
        return response


def get_dataset(name, path):
    if name == 'ml-1m':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def u_m_mapping(dataset, train_dataset_index):
    u_traindata = {}
    u_traindata_pos = {}
    u_traindata_neg = {}
    m_traindata = {}
    u_m_traindata = {}
    m_u_traindata = {}
    u_m_targets = {}
    m_u_targets = {}
    u_m_time = {}
    for index in train_dataset_index:
        u_id = dataset.items[index][0]
        m_id = dataset.items[index][1]
        m_traindata[m_id] = dataset.m_data[m_id]
        if u_id not in list(u_m_traindata.keys()):
            u_m_traindata[u_id] = []
        u_m_traindata[u_id].append(m_id)
        if m_id not in list(m_u_traindata.keys()):
            m_u_traindata[m_id] = []
        m_u_traindata[m_id].append(u_id)
        u_m_targets[str(u_id) + '_' + str(m_id)] = dataset.targets[index]
        m_u_targets[str(m_id) + '_' + str(u_id)] = dataset.targets[index]
        u_m_time[str(u_id) + '_' + str(m_id)] = dataset.time[index]
        if dataset.targets[index] == 1.0:
            if u_id not in list(u_traindata_pos.keys()):
                u_traindata_pos[u_id] = []
            u_traindata_pos[u_id].append(dataset.m_data[m_id])
        if dataset.targets[index] == 0.0:
            if u_id not in list(u_traindata_neg.keys()):
                u_traindata_neg[u_id] = []
            u_traindata_neg[u_id].append(dataset.m_data[m_id])
    u_traindata['pos'] = u_traindata_pos
    u_traindata['neg'] = u_traindata_neg
    return u_traindata, m_traindata, u_m_traindata, m_u_traindata, u_m_targets, m_u_targets, u_m_time


def select_sample(i, min_idx, clustered_u_idx, clustered_sentences, u_m_traindata, u_m_targets, u_m_time, type):
    select_u_idx = [clustered_u_idx[i][j] for j in min_idx]
    select_u_text = [clustered_sentences[i][j] for j in min_idx]
    select_u_like_m_idx = []
    select_u_dislike_m_idx = []
    if type == 'item':
        for k in select_u_idx:
            u_like_m_per_user = []
            u_dislike_m_per_user = []
            m_id = u_m_traindata[k]
            for m in m_id:
                if u_m_targets[str(k) + '_' + str(m)] == 1.0:
                    u_like_m_per_user.append(m)
                else:
                    u_dislike_m_per_user.append(m)
            u_like_m_per_user_time = [u_m_time[str(k) + '_' + str(m)] for m in u_like_m_per_user]
            u_like_m_per_user_time = sorted(range(len(u_like_m_per_user_time)), key=lambda k: u_like_m_per_user_time[k],
                                            reverse=True)
            u_like_m_per_user = [u_like_m_per_user[i] for i in
                                 u_like_m_per_user_time[:int(len(u_like_m_per_user_time))]]
            select_u_like_m_idx += u_like_m_per_user

            u_dislike_m_per_user_time = [u_m_time[str(k) + '_' + str(m)] for m in u_dislike_m_per_user]
            u_dislike_m_per_user_time = sorted(range(len(u_dislike_m_per_user_time)),
                                               key=lambda k: u_dislike_m_per_user_time[k],
                                               reverse=True)
            u_dislike_m_per_user = [u_dislike_m_per_user[i] for i in
                                    u_dislike_m_per_user_time[:int(len(u_dislike_m_per_user_time))]]
            select_u_dislike_m_idx += u_dislike_m_per_user
    else:
        for k in select_u_idx:
            m_id = u_m_traindata[k]  # use timestamp to select new m_id for each k.
            for m in m_id:
                if u_m_targets[str(k) + '_' + str(m)] == 1.0:
                    select_u_like_m_idx.append(m)
                else:
                    select_u_dislike_m_idx.append(m)

    demo_element = {
        "user_id": select_u_idx,
        "user_info": set(select_u_text),
        "user_like_movies": set(select_u_like_m_idx),
        "user_dislike_movies": set(select_u_dislike_m_idx)
    }
    return demo_element


def clustering(u_id, u_text, u_m_traindata, u_m_targets, encoder, num_clusters, seed, topk, u_text_test, u_m_time,
               num_sets, type='user'):
    u_text_emb = encoder.encode(u_text)
    clustering_model = KMeans(n_clusters=num_clusters, random_state=seed)
    clustering_model.fit(u_text_emb)
    cluster_assignment = clustering_model.labels_

    u_text_test_emb = encoder.encode(u_text_test)
    cluster_assignment_test = clustering_model.predict(u_text_test_emb)

    clustered_sentences = [[] for i in range(num_clusters)]
    dist = clustering_model.transform(u_text_emb)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_u_idx = [[] for i in range(num_clusters)]
    for user_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(u_text[user_id])
        clustered_dists[cluster_id].append(dist[user_id][cluster_id])
        clustered_u_idx[cluster_id].append(u_id[user_id])

    demos = []
    for i in range(len(clustered_dists)):
        print("Cluster ", i + 1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        idx = list(range(len(clustered_dists[i])))
        demo_elements = []
        for num_set in range(num_sets):
            idx_topk = idx[topk * num_set:topk * (num_set + 1)]
            min_idx = [top_min_dist[i][0] for i in idx_topk]
            demo_element = select_sample(i, min_idx, clustered_u_idx, clustered_sentences, u_m_traindata, u_m_targets,
                                         u_m_time, type)
            demo_elements.append(demo_element)
        demos.append(demo_elements)

    return demos, cluster_assignment_test


def individual_test_info(u_id_test, u_m_traindata, u_m_targets):
    u_test_reformat = {}
    for u in u_id_test:
        u_like_m = []
        u_dislike_m = []
        if u in list(u_m_traindata.keys()):
            m_history = u_m_traindata[u]
            for m in m_history:
                if u_m_targets[str(u) + '_' + str(m)] == 1.0:
                    u_like_m.append(m)
                else:
                    u_dislike_m.append(m)
        u_test_reformat_element = {'like': set(u_like_m), 'dislike': set(u_dislike_m)}
        u_test_reformat[u] = u_test_reformat_element
    return u_test_reformat


def counter(item, prefix=None):
    counter = collections.Counter(item)
    if len(set(item)) != 2:
        list_prefer = counter.most_common(int(len(set(item)) / 2))
    else:
        list_prefer = counter.most_common(2)
    for i in range(len(list_prefer)):
        list_prefer[i] = str(round(list_prefer[i][1] / len(item) * 100, 1)) + '% ' + \
                         list_prefer[i][0]
        # if i == 0:
        #     list_prefer[i] = str(round(list_prefer[i][1] / len(item) * 100, 1)) + "% of the user's preferred" + prefix + ' are ' + list_prefer[i][0]
        # else:
        #     list_prefer[i] = str(round(list_prefer[i][1] / len(item) * 100, 1)) + '% are ' + \
        #                      list_prefer[i][0]
    return list_prefer


def aggregate_info(u_like_m_info, u_like_m_time=None, type='user', demon=False):
    u_like_m_title = u_like_m_info
    # u_like_m_genres = []
    # u_like_m_year = []
    # for info in u_like_m_info:
    #     u_like_m_title += [info]
        # if isinstance(info[1], list):
        #     u_like_m_genres += info[1]
        # else:
        #     u_like_m_genres += [info[1]]
        # u_like_m_year += [info[2]]
    if type == 'user':
        u_like_m_title = ', '.join(counter(u_like_m_title, 'genders'))
        # u_like_m_genres = ', '.join(counter(u_like_m_genres, 'ages'))
        # u_like_m_year = ', '.join(counter(u_like_m_year, 'occupations'))
    else:
        if not demon:
            if len(u_like_m_time) < 50:
                u_like_m_title = [u_like_m_title[i] for i in u_like_m_time[:int(len(u_like_m_time))]]
            else:
                u_like_m_title = [u_like_m_title[i] for i in u_like_m_time[:int(len(u_like_m_time) / 2)]]
            # u_like_m_year = [u_like_m_year[i] for i in u_like_m_time[:int(len(u_like_m_time) / 10)]]
        u_like_m_title = ', '.join(u_like_m_title)
        # u_like_m_genres = ', '.join(counter(u_like_m_genres, 'genres'))
        # u_like_m_year = ', '.join(set(u_like_m_year))
    return u_like_m_title


# def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
#     vote_results = [0] * n_candidates
#     for vote_output in vote_outputs:
#         if 'yes' in vote_output.lower():
#             vote_results[0] += 1
#         else:
#             vote_results[1] += 1
#     if vote_results[0] > vote_results[1]:
#         return 'yes'
#     else:
#         return 'no'
def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
    vote_results = [0] * n_candidates
    for vote_output in vote_outputs:
        pattern = r".*best choice is .*(\d+).*"
        match = re.match(pattern, vote_output, re.DOTALL)
        if match:
            vote = int(match.groups()[0]) - 1
            if vote in range(n_candidates):
                vote_results[vote] += 1
        else:
            print(f'vote no match: {[vote_output]}')
    return vote_results


def main(dataset_name,
         dataset_path,
         seed,
         batch_size,
         encoder,
         topk,
         temperature,
         n_generate_sample,
         method_evaluate,
         model,
         num_clusters,
         steps,
         num_sets):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    fix_seed(seed)
    encoder = SentenceTransformer(encoder)
    
    u_traindata, m_traindata, u_testdata, m_testdata, u_m_testdata, u_m_targets_test, \
    u_id_test, u_text_test, u_demons, cluster_assignment_test_u, m_id_test, m_text_test, \
    u_test_reformat, u_m_time, u_m_time_test = pickle.load(
        open(dataset_name + 'processed.pkl', 'rb'))

    gt = []
    preds = []
    preds_or = []
    count = 0
    count_or = 0
    test_uid = []
    for i in range(len(u_id_test)):# 121
        # if i in [7, 24, 27]:
        #     continue
        uid = u_id_test[i]
        test_uid.append(uid)
        u_info = u_text_test[i]
        if uid in list(u_test_reformat.keys()):
            prompt_strs = [0]
            prompt_strs1 = [0]
            u_like_m_id = u_test_reformat[uid]['like']
            u_like_m_info = [m_traindata[m] for m in u_like_m_id]
            u_like_m_time = [u_m_time[str(uid) + '_' + str(m)] for m in u_like_m_id]
            u_like_m_time = sorted(range(len(u_like_m_time)), key=lambda k: u_like_m_time[k], reverse=True)
            u_like_m_title = aggregate_info(u_like_m_info, u_like_m_time, type='item')
            str1 = "the recent appliances this user likes are " + u_like_m_title + '.'

            u_dislike_m_id = u_test_reformat[uid]['dislike']
            u_dislike_m_info = [m_traindata[m] for m in u_dislike_m_id]
            u_dislike_m_time = [u_m_time[str(uid) + '_' + str(m)] for m in u_dislike_m_id]
            u_dislike_m_time = sorted(range(len(u_dislike_m_time)), key=lambda k: u_dislike_m_time[k], reverse=True)
            u_dislike_m_title= aggregate_info(u_dislike_m_info, u_dislike_m_time, type='item')
            if u_dislike_m_title=='':
                str2 = ''
            else:
                str2 = 'the recent appliances this user does not like are ' + u_dislike_m_title + '. '

        else:
            str1 = ''
            str2 = ''
        cot_prompt = '''
        Briefly summarize the user's appliance preference based on the following information:

        {input}

        Only output several short sentences in the following format without any additional text or thoughts! And each sentence has no redundancy with other sentences:
        - The user likes appliances that consume less energy while maintaining high performance.
        - The user likes appliances that are easy to clean and maintain.
        - The user likes appliances with user-friendly interfaces and intuitive controls.
        - The user likes quieter appliances.
        '''
        cot_prompt_1 = '''
        Further summarize the user's appliance preference based on the following two information:

        The user's appliance preference we have already:
        {input1}

        Appliance history:
        {input2}

        Only output several short sentences in the following format without any additional text or thoughts! Each sentence has no redundancy with other sentences, and do not repeat the above preference we have already:
        - The user doesn't like cheap appliances.
        - The user is neutral to appliances that come with some noise.
        - The user doesn't like appliances occupying too much space.
        '''
        cot_prompt_2 = '''
        Briefly summarize the appliance preference of similar users based on the following information:

        {input}

        Only output several short sentences in the following format without any additional text or thoughts! And each sentence has no redundancy with other sentences:
        - They like appliances that consume less energy while maintaining high performance.
        - They like appliances that are easy to clean and maintain.
        - They like appliances with user-friendly interfaces and intuitive controls.
        - They like quieter appliances.
        '''

        cot_prompt_3 = '''
        Further summarize the appliance preference of similar users based on the following two information:

        The appliance preference of similar users we have already:
        {input1}

        Appliance history:
        {input2}

        Only output several short sentences in the following format without any additional text or thoughts! Each sentence has no redundancy with other sentences, and do not repeat the above preference we have already:
        - They don't like cheap appliances.
        - They are neutral to appliances that come with some noise.
        - They don't like appliances occupying too much space.
        '''

        prompt_strs[0] = str1
        prompt_strs1[0] = str2
        u_c_id = cluster_assignment_test_u[i]
        u_demon = u_demons[u_c_id]

        for demon_id in range(len(u_demon)):
            demon = u_demon[demon_id]
            u_demon_like_id = demon["user_like_movies"]
            u_demon_like_info = [m_traindata[m] for m in u_demon_like_id]
            u_like_m_title = aggregate_info(u_demon_like_info, type='item', demon=True)
            str3 = 'the recent appliances similar users like are ' + u_like_m_title + '.'

            u_demon_dislike_id = demon["user_dislike_movies"]
            u_demon_dislike_info = [m_traindata[m] for m in u_demon_dislike_id]
            u_dislike_m_title = aggregate_info(u_demon_dislike_info, type='item', demon=True)
            str4 = 'the recent appliances similar users do not like are ' + u_dislike_m_title + '. '
            prompt_strs.append(str3)
            prompt_strs1.append(str4)

        ys = ''
        ##### summary for user
        for step in range(len(prompt_strs)):
            if step == 0:
                prompt = cot_prompt.format(input=prompt_strs[step])
                samples = gpt(prompt, model=model, temperature=0, max_tokens=1000, n=1)
                ys += '''This user's preference on appliances:\n''' + samples[0] + '\n\n'
                if prompt_strs1[step] != '':
                    prompt = cot_prompt_1.format(input1=samples[0], input2=prompt_strs1[step])
                    samples = gpt(prompt, model=model, temperature=0, max_tokens=1000, n=1)
                    ys += samples[0] + '\n\n'
            else:
                old_ys = ys
                prompt = cot_prompt_2.format(input=prompt_strs[step])
                samples = gpt(prompt, model=model, temperature=0, max_tokens=1000, n=1)
                ys += '''Some similar users' preference on appliances:\n''' + samples[0] + '\n\n'
                prompt = cot_prompt_3.format(input1=samples[0], input2=prompt_strs1[step])
                samples = gpt(prompt, model=model, temperature=0, max_tokens=1000, n=1)
                ys += samples[0] + '\n\n'
                #### early stop if ys similar to ys_old

                embeddings_ys = encoder.encode([old_ys, ys])
                similarity_ys = cosine_similarity([embeddings_ys[0]], [embeddings_ys[1]])[0][0]
                if similarity_ys > 0.95:
                    break
        # ys = ys_0 + ys

        m_ids = u_m_testdata[uid]  # m_ids[2],m_ids[3],
        for m in m_ids:
            index = m_id_test.index(m)
            m_info = m_text_test[index]

            score_prompt = '''Conclude how likely the user likes the appliance with an exact probabilty number, the number should range from 0 to 1, 0 means 'totally dislike' and 1 means 'totally like', do not explain the reason and include any other words.'''

            prompt = ys + 'Given an appliance titled ' + m_info + '. ' + score_prompt
            # prompt_m = ys_m + u_info + score_prompt
            pred = gpt(prompt, model=model, temperature=1, max_tokens=10, n=5)
            # pred_m = gpt(prompt_m, model=model, temperature=1, max_tokens=10, n=5)
            print('prompt:', prompt)
            # print('prompt_m:', prompt_m)
            print('user prediction is:', pred)
            # print('movie prediction is:', pred_m)
            print('gt is:', u_m_targets_test[str(uid) + '_' + str(m)])
            preds.append(pred)
            gt.append(u_m_targets_test[str(uid) + '_' + str(m)])
    pickle.dump((gt, preds), open(dataset_name + 'res_with_vote_total.pkl', 'wb'))
    pickle.dump(test_uid, open(dataset_name + '_test_uid_total.pkl', 'wb'))
            # if pred[0] not in ['1', '0']:
            #     pred[0] = '0'
            #     print('replaced user prediction with 0')
            # if pred_m[0] not in ['1', '0']:
            #     pred_m[0] = '0'
            #     print('replaced movie prediction with 0')

    #         if float(pred[0]) == float(pred_m[0]) and float(pred[0]) == 1.0:
    #             pred_final = 1.0
    #         else:
    #             pred_final = 0.0
    #
    #         if float(pred[0]) == 1.0 or float(pred_m[0]) == 1.0:
    #             pred_final_or = 1.0
    #         else:
    #             pred_final_or = 0.0
    #
    #         gt.append(u_m_targets_test[str(uid) + '_' + str(m)])
    #         preds.append(pred_final)
    #         preds_or.append(pred_final_or)
    #         if pred_final_or == u_m_targets_test[str(uid) + '_' + str(m)]:
    #             count_or += 1
    #         if pred_final == u_m_targets_test[str(uid) + '_' + str(m)]:
    #             count += 1
    # # auc = roc_auc_score(gt, preds)
    # # auc_or = roc_auc_score(gt, preds_or)
    # # print('auc: ', auc)
    # print('acc: ', count / 68)
    # # print('auc_or: ', auc_or)
    # print('acc_or: ', count_or / 68)
    # count_all_pos = 0
    # count_all_neg = 0
    # for i in range(100):
    #     count_pos = 0
    #     count_neg = 0
    #     uid = u_id_test[i]
    #     m_ids = u_m_testdata[uid]
    #     for m in m_ids:
    #         if u_m_targets_test[str(uid) + '_' + str(m)] == 1.0:
    #             count_pos += 1
    #         else:
    #             count_neg += 1
        # print('each i:\n')
    #     print(i, count_pos, count_neg)
    #     count_all_pos += count_pos
    #     count_all_neg += count_neg
        # print('top i:\n')
        # print(count_all_pos, count_all_neg)
    # print('all:\n')
    # print(count_all_pos, count_all_neg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='amazon-appliance')
    parser.add_argument('--dataset_path', default='amazon-reviews/')
    parser.add_argument('--seed', default=192)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--encoder', type=str, default="all-MiniLM-L6-v2",
                        help="which sentence-transformer encoder for clustering")
    parser.add_argument('--topk', default=5)
    parser.add_argument('--temperature', default=1.)
    parser.add_argument('--n_generate_sample', default=3)
    parser.add_argument('--method_evaluate', default='vote')
    parser.add_argument('--model', type=str, default="gpt-4"
                                                     "", choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-4-1106-preview'])
    parser.add_argument('--num_clusters', default=5)
    parser.add_argument('--steps', default=2)
    parser.add_argument('--num_sets', default=5)

    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.seed,
         args.batch_size,
         args.encoder,
         args.topk,
         args.temperature,
         args.n_generate_sample,
         args.method_evaluate,
         args.model,
         args.num_clusters,
         args.steps,
         args.num_sets
         )
