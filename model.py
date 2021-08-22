import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import dgl
import dgl.nn as dglnn


class NodeAttention(layers.Layer):
    def __init__(self, out_feat, hidden_feat=128, num_heads=1):
        super(NodeAttention, self).__init__()
        self.gat_conv = dglnn.GATConv(
            hidden_feat, out_feat, num_heads, activation=keras.activations.elu, allow_zero_in_degree=True)

    def call(self, graph, h):
        z = self.gat_conv(graph, h)
        z = tf.reshape(z, [z.shape[0], z.shape[1]*z.shape[2]])
        return z


class SemanticAttention(layers.Layer):
    def __init__(self, hidden_feat=128):
        super(SemanticAttention, self).__init__()
        self.model = keras.Sequential([
            layers.Dense(hidden_feat, activation='tanh', use_bias=True),
            layers.Dense(1, use_bias=False),
        ])

    def call(self, inputs):
        """Return the weight for the meta_path

        :param inputs: tensor with shape (V, D). 
        (V is the size of nodes. D is the node feature dimension.)

        :return w_meta_p: the weight for the meta_path
        """
        w = self.model(inputs)  # (V, 1)
        w_meta_p = tf.math.reduce_mean(w, 0)  # (1)
        return w_meta_p


class HANLayer(layers.Layer):
    def __init__(self, ntypes, meta_paths, out_feat=128, hidden_feat=128, num_heads=10):
        super(HANLayer, self).__init__()
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.project_layers = {
            node: layers.Dense(hidden_feat, use_bias=False) for node in ntypes
        }
        self.node_attention_layers = {
            meta_path: NodeAttention(out_feat, hidden_feat, num_heads) for meta_path in meta_paths
        }
        self.semantic_attention_layer = SemanticAttention(
            hidden_feat=hidden_feat)

    def call(self, graph, feat):
        """
        """
        h = {node: self.project_layers[node](feat[node]) for node in feat}
        embeddings = {node: [] for node in feat}
        w = []  # (M, 1)
        for src, meta_path, dst in graph.canonical_etypes:
            z = self.node_attention_layers[meta_path](
                graph[meta_path], (h[src], h[dst]))
            for node in graph.ntypes:
                if node != dst:
                    embeddings[dst].append(
                        tf.zeros([graph.num_nodes(dst), self.num_heads*self.out_feat]))
                else:
                    embeddings[dst].append(z)
            w.append(self.semantic_attention_layer(z))
        for node in feat:
            embeddings[node] = tf.stack(embeddings[node], axis=1)
        beta = tf.nn.softmax(w, axis=0)  # (M, 1)
        for node in feat:
            embeddings[node] = tf.reduce_sum(beta*embeddings[node], axis=1)
        return embeddings

class HAN(keras.Model):
    def __init__(self, ntypes, meta_paths, out_feat=128, hidden_feat=128, num_heads=10):
        super(HAN, self).__init__()
        self.layer1 = HANLayer(ntypes, meta_paths, out_feat=out_feat,
                               hidden_feat=hidden_feat, num_heads=num_heads)
        self.layer2 = layers.Dense(2,activation='softmax')
    
    def call(self, graph, feat, pick):
        h = self.layer1(graph, feat)
        h = h[pick]
        h = self.layer2(h)
        return h
