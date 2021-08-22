import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import dgl
from dgl.data.utils import load_graphs
from model import HAN
import time

# Get model and make a simple test
# g = dgl.heterograph({
#     ('drug', 'interacts', 'drug'): (tf.constant([0, 1]), tf.constant([1, 2])),
#     ('drug', 'interact', 'gene'): (tf.constant([0, 1]), tf.constant([2, 3])),
#     ('drug', 'treats', 'disease'): (tf.constant([1]), tf.constant([2]))
# })
# feat = {
#     'drug': tf.random.uniform([3, 5]),
#     'gene': tf.random.uniform([4, 6]),
#     'disease': tf.random.uniform([3, 7])
# }
# y = tf.constant([1,1,0])
# g = g.to('/gpu:0')


g, feat = load_graphs("./data/graph/heterograph1.bin")
g = g[0]
y = feat.pop('label')
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_nid_dict={}
for node in g.ntypes:
    train_nid_dict[node]=tf.constant(range(g.num_nodes(node)))
    
print(train_nid_dict)
dataloader = dgl.dataloading.NodeDataLoader(
    g, train_nid_dict, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)
model = HAN(g.ntypes, g.etypes)

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

epochs = 10
for epoch in range(epochs):
    print("<---- Start epoch %d: ----->"%(epoch))

    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to('/gpu:0') for b in blocks]
        input_features = blocks[0].srcdata     # returns a dict
        output_labels = blocks[-1].dstdata     # returns a dict
        with tf.GradientTape() as tape:
            logits = model(blocks, input_features, 'user', training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)

        print(
            "training loss (for one batch): %.4f"
            % (float(loss_value))
        )
    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #     val_logits = model(x_batch_val, training=False)
    #     # Update val metrics
    #     val_acc_metric.update_state(y_batch_val, val_logits)
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print("Validation acc: %.4f" % (float(val_acc),))
    # print("Time taken: %.2fs" % (time.time() - start_time))