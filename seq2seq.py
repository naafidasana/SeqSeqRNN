import torch
import torch.nn as nn
import utils
from modules import Encoder, Decoder, EncoderDecoder, MaskedSoftmaxCELoss


class Seq2SeqEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # The output 'X' shape: ('batch_size', 'num_steps', 'embed_size')
        X = self.embedding(X)
        # First axis corresponds to time steps since this is an RNN model
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zero.
        output, state = self.rnn(X)
        # 'output' shape: ('num_steps', 'batch_size', 'num_hiddens')
        # 'state' shape: ('num_layers', 'batch_size', 'num_hiddens')
        return output, state


class Seq2SeqDecoder(Decoder):
    """The RNN decoder for sequence to sequence learning."""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens,
                          num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output 'X' shape: ('batch_size', 'num_steps', 'embed_size')
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast 'context' so it has the same 'num_steps' as 'X'.
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for Sequence to Sequence."""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # The softmax cross-entropy loss with masks, implemented in modules
    loss = MaskedSoftmaxCELoss()
    net.train()

    accumulator = {'l': 0, 'num_tokens': 0}

    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor(
                [tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            utils.gradient_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                accumulator['l'], accumulator['num_tokens'] = l.sum(
                ), num_tokens
        if (epoch + 1) % 1 == 0:
            print(
                f"loss: {accumulator['l']/accumulator['num_tokens']:.3f}")


embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, utils.try_gpu()
filepath = "dt1_update.tsv"
train_iter, src_vocab, tgt_vocab = utils.load_nmt_data(
    filepath, batch_size, num_steps)


encoder = Seq2SeqEncoder(len(src_vocab), embed_size,
                         num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size,
                         num_hiddens, num_layers, dropout)
net = EncoderDecoder(encoder, decoder)

train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# Test model performance after training
dag = ['m ma n bala', 'O bɔri alabalaayi bilibu pam', 'o kanna kpe biɛɣukam']
eng = ['that is my mom', 'She likes inciting trouble', 'he comes here everyday']
for dag, eng in zip(dag, eng):
    translation, attention_weight_seq = utils.predict_seq2seq(
        net, dag, src_vocab, tgt_vocab, num_steps, device,
    )
    print(f"{dag} => {translation}, bleu {utils.bleu(translation, eng, k=2):.3f}")
