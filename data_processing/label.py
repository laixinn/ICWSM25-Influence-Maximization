#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import igraph as ig
import json, codecs
import pandas as pd
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
from tqdm import tqdm
from itertools import count, islice
import pickle5 as pickle


# In[2]:


class BaseClass:
    def __init__(self):
        self.graph_path = 'tweet_week/'
        self.motif_path = 'tweet_motif/motif.csv'
        self.label_path = 'tweet_week/'
        self.week_dts = []
        self.init_week_dts()

    def init_week_dts(self):
        file_list = os.listdir(self.graph_path)
        for file_name in file_list:
            if file_name.endswith("_graph.csv"): 
                time_str = file_name.split("_")[0] 
                self.week_dts.append(int(time_str))
        self.week_dts = sorted(self.week_dts)


# In[3]:


class SparkDataFrameHandler(BaseClass):
    def __init__(self):
        super().__init__()

    def load_week_strong(self, t):
        filename = os.path.join(self.graph_path, f'%s_strong.csv' %t)
        df = pd.read_csv(filename, usecols=['role_id', 't_role_id', 'dt', 'real_dt'])
        return df

    def load_week_weak(self, t):
        filename = os.path.join(self.graph_path, f'%s_graph.csv' %t)
        df = pd.read_csv(filename, usecols=['role_id', 't_role_id', 'dt', 'real_dt'])
        return df

    def load_data(self, cls='strong'):
        print('##### LOADING DATA... #####')
        graph = pd.DataFrame()
        for dt in self.week_dts:
            if cls == 'strong':
                week_graph = self.load_week_strong(dt)
            else:
                week_graph = self.load_week_weak(dt)
                # print(len(week_graph[['role_id', 't_role_id']].stack().value_counts()))
            graph = pd.concat([graph, week_graph], axis=0)
        graph = graph.reset_index(drop=True)
        graph = graph.rename(columns={'dt': 'week_dt'})
        graph = graph.rename(columns={'real_dt': 'dt'})
        return graph
    
    def load_motif(self):
        df = pd.read_csv(self.motif_path)
        return df
    
    def load_label(self, name):
        filename = self.label_path+name+'.csv'
        df = pd.read_csv(filename)
        return df
    
    def save_motif(self, df):
        if not os.path.exists('tweet_motif'):
            os.makedirs('tweet_motif')
        df.to_csv(self.motif_path, index=False)

    def save_label(self, df, name):
        filename = self.label_path+name+'.csv'
        df.to_csv(filename)
    


# In[4]:


class GraphHandler:
    def __init__(self):
        return
    
    def read_graph(self, date):
        filename = f"week_graph/week_g_{date}.gpickle"
        # graph_date = nx.read_gpickle(filename)
        with open(filename, 'rb') as f:
            graph_date = pickle.load(f)
        return graph_date

    def convert_graph(self, ig_graph):
        # 将igraph转换为networkx图
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(ig_graph.get_edgelist())
        return nx_graph
    


# In[5]:


class GraphConstructor(BaseClass):
    def __init__(self):
        super().__init__()
        self.all_nodes = {}
        self.node_num = 0

    def read_weekly_graph(self, week_date):
        path = os.path.join('week_graph', f'week_g_{week_date}.gpickle')
        weekly_graph = ig.read(path, 'pickle')
        print(f'graph {week_date} is loaded.')
        return weekly_graph

    def construct_weekly_graph(self, data_frame):
        graph = ig.Graph()
        edges = []

        for index, item in data_frame.iterrows():
            role_id = item['role_id']
            target_role_id = item['t_role_id']

            if role_id not in self.all_nodes:
                self.all_nodes[role_id] = self.node_num
                self.node_num += 1
            if target_role_id not in self.all_nodes:
                self.all_nodes[target_role_id] = self.node_num
                self.node_num += 1

            edges.append((self.all_nodes[role_id], self.all_nodes[target_role_id]))

        graph.add_vertices(self.node_num)
        graph.add_edges(edges)

        if not os.path.exists('week_graph'):
            os.makedirs('week_graph')

        save_path = os.path.join('week_graph', f'week_g_{item["week_dt"]}.gpickle')
        graph.write_pickle(save_path)

    def save_all_nodes(self):
        self.all_nodes = {int(key): value for key, value in self.all_nodes.items()}

        if not os.path.exists('week_graph'):
            os.makedirs('week_graph')
        all_nodes_path = 'week_graph/all_nodes.json'
        with codecs.open(all_nodes_path, 'w', 'utf8') as outf:
            json.dump(self.all_nodes, outf, ensure_ascii=False)
            outf.write('\n')
        print(f'{all_nodes_path} saved.')

    def construct_graph(self, weekly_data_frame):
        # week_dates = weekly_data_frame.week_dt.drop_duplicates().reset_index(drop=True).sort_values()

        for week_date in self.week_dts:
            week_data_frame = weekly_data_frame[weekly_data_frame['week_dt'] == week_date].reset_index(drop=True)
            self.construct_weekly_graph(week_data_frame)

        self.save_all_nodes()




class MotifMiner(BaseClass):
    def __init__(self):
        super().__init__()
        self.total_motif = pd.DataFrame()

    def executor_generator(self, nx_graph):
        return GrandIsoExecutor(graph=nx_graph)

    def motif_line_generator(self):
        line = Motif("""
            A -> B
            B -> C
            """,
            ignore_direction=True,
            exclude_automorphisms=True)
        return line
    
    def motif_triangle_generator(self):
        line = Motif("""
            A -> B
            B -> C
            C -> A
            """,
            ignore_direction=True,
            exclude_automorphisms=True)
        return line

    def get_motif_subgraphs(self, week_dates):
        for _, date in enumerate(tqdm(week_dates)):
            ig_graph_date = GraphHandler().read_graph(date)
            nx_graph_date = GraphHandler().convert_graph(ig_graph_date)
            executor = self.executor_generator(nx_graph_date)
            motif = self.motif_line_generator()
            results = executor.find(motif)

            result_df = pd.DataFrame(results)
            result_df['week_dt'] = date
            self.total_motif = pd.concat([self.total_motif, result_df], axis=0)
            
        self.total_motif = self.total_motif.reset_index(drop=True)
        self.total_motif = self.total_motif.astype(int)
        SparkDataFrameHandler().save_motif(self.total_motif)
            
    def motif_mining(self):
        self.get_motif_subgraphs(self.week_dts)


# In[7]:


class InfluencerLabeler(BaseClass):
    def __init__(self):
        super().__init__()
        self.total_label = pd.DataFrame()
        self.total_cold = pd.DataFrame()

    def drop_duplicate_motifs(self, df):
        df['tuple'] = df.apply(lambda row: tuple(sorted(row)), axis=1)
        df = df.drop_duplicates(subset='tuple').reset_index(drop=True)
        df = df.drop(columns='tuple')
        return df

    def read_all_nodes(self):
        with open('week_graph/all_nodes.json', 'r') as f:
            all_nodes = json.load(f)
        flipped_dict = {value: key for key, value in all_nodes.items()}
        return flipped_dict

    def map_uid(self, df):
        flipped_dict = self.read_all_nodes()
        df['A'] = df['A'].map(flipped_dict)
        df['B'] = df['B'].map(flipped_dict)
        df['C'] = df['C'].map(flipped_dict)
        
        return df
    
    def filter_motifs(self, t1_motifs_df, max_t_nodes, t_rec_nodes):
        mask = (t1_motifs_df['A'] < max_t_nodes) & (t1_motifs_df['B'] > max_t_nodes)\
              & (t1_motifs_df['C'] > max_t_nodes)
        new_df = t1_motifs_df[mask].drop_duplicates().reset_index(drop=True)
    
        new_df = self.drop_duplicate_motifs(new_df)
        new_df = self.map_uid(new_df)
        new_df['A'] = new_df['A'].astype(int)
        new_df = new_df[new_df['A'].isin(t_rec_nodes)].drop_duplicates().reset_index(drop=True)

        return new_df
    
    def filter_time(self, dt1, new_df, total_weekly_df):
        # initialization
        week_df = total_weekly_df[total_weekly_df['week_dt']==dt1].reset_index(drop=True)
        new_df[['A', 'B', 'C', 'week_dt']] = new_df[['A', 'B', 'C', 'week_dt']].astype(int)
        week_df[['role_id', 't_role_id']] = week_df[['role_id', 't_role_id']].astype(int)

        # find interaction time of A & B in week_df
        df_AB = self.find_interaction_time(week_df, new_df, 'A', 'B')
        df_BC = self.find_interaction_time(week_df, df_AB, 'B', 'C')

        df_fin = df_BC.drop_duplicates().reset_index(drop=True)
        df_fin = df_fin[df_fin.dt_AB <= df_fin.dt_BC].reset_index(drop=True)

        return df_fin
    
    def find_interaction_time(self, week_df, df, col1_name, col2_name):
        week_df_12 = week_df.rename(columns={'role_id': col1_name, 't_role_id': col2_name})
        week_df_21 = week_df.rename(columns={'role_id': col2_name, 't_role_id': col1_name})
        df1 = df.merge(week_df_12, on=[col1_name, col2_name], how='left').dropna()
        df2 = df.merge(week_df_21, on=[col1_name, col2_name], how='left').dropna()
        df3 = pd.concat([df1, df2], axis=0).drop_duplicates().reset_index(drop=True)
        df3 = df3.rename(columns={'dt': 'dt_%s' %(col1_name+col2_name)})
        return df3

    def label(self, dt, df_fin, total_weekly_df):
        df_fin['A'] = df_fin['A'].astype(str)
        influencer = df_fin['A'].drop_duplicates().reset_index(drop=True).astype(str).tolist()

        label_df = pd.DataFrame(columns=['role_id', 'label', 'week_dt'])
        label_week_df = total_weekly_df[total_weekly_df['week_dt']==dt].reset_index(drop=True)
        week_ids = pd.concat([label_week_df['role_id'], label_week_df['t_role_id']]).astype(str).drop_duplicates().reset_index(drop=True)

        label_df['role_id'] = week_ids
        label_df['label'] = 0
        label_df['week_dt'] = dt
        label_df.loc[label_df['role_id'].isin(influencer), 'label'] = 1

        SparkDataFrameHandler().save_label(label_df, '%s_labels'%dt)

    def find_influencers(self, week_dates, total_weekly_df):
        motifs_df = SparkDataFrameHandler().load_motif()
        total_rec_week_df = SparkDataFrameHandler().load_data(cls='weak')

        for i in range(len(week_dates)-1):
            dt = week_dates[i]
            dt1 = week_dates[i+1]
            
            t_g = GraphHandler().read_graph(dt)
            t_g = GraphHandler().convert_graph(t_g)
            t_nodes = t_g.nodes()
            max_t_nodes = max(t_nodes)

            # get t1 motifs
            t1_motifs_df = motifs_df[motifs_df['week_dt']==dt1]
            
            # get nodes in rec in previous week
            t_rec_df = total_rec_week_df[total_rec_week_df['week_dt']==dt]
            t_rec_nodes = pd.unique(t_rec_df[['role_id', 't_role_id']].values.ravel()).tolist()

            # filter out motifs in old-new-new
            new_df = self.filter_motifs(t1_motifs_df, max_t_nodes, t_rec_nodes)

            # filter time
            df_fin = self.filter_time(dt1, new_df, total_weekly_df)
            label_len = len(df_fin['A'].drop_duplicates().reset_index(drop=True))
            print("{:.3f}".format(label_len/len(t_nodes)), label_len, '/', len(t_nodes))
            
            #label
            self.label(dt, df_fin, total_weekly_df)

            # cold
            df_fin['week_dt_AB'] = dt
            df_fin['week_dt_BC'] = dt1
            self.total_cold = pd.concat([self.total_cold, df_fin[['A', 'B', 'C', 'week_dt_AB', 'week_dt_BC']]], axis=0)

        self.total_cold = self.total_cold.reset_index(drop=True)
        SparkDataFrameHandler().save_label(self.total_cold, 'path')

            
        # self.total_label = self.total_label.reset_index(drop=True)
        # SparkDataFrameHandler().save_label(self.total_label)
            
    def final_labeling(self, total_weekly_df):
        self.find_influencers(self.week_dts, total_weekly_df)


# In[8]:


def main():
    # reading weekly table
    handler = SparkDataFrameHandler()
    total_weekly_df = handler.load_data(cls='strong')
    # graph construction
    gc = GraphConstructor()
    gc.construct_graph(total_weekly_df)
    print('Construct graph success')
    # motif mining
    miner = MotifMiner()
    miner.motif_mining()
    print('Motif mining success')
    # labeling
    labeler = InfluencerLabeler()
    total_weekly_weak_df = handler.load_data(cls='weak')
    labeler.final_labeling(total_weekly_weak_df)



if __name__ == '__main__':
    main()






