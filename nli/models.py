import torch
import torch.nn as nn

class AvgWordEmb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, length):

        assert embedding.shape == (length.shape[0], 100, 300)

        mask = (length.unsqueeze(1) > torch.arange(embedding.shape[1], device=embedding.device)).float().unsqueeze(2)
        mask = mask.to(embedding.device)

        embedding_sum = torch.sum(embedding * mask, dim = 1)
        length_sum = torch.sum(mask, dim = 1)
        mean = embedding_sum / length_sum

        return mean


class UniLSTM(torch.nn.Module):
    """
    Unidirectional LSTM applied on the word embeddings, where the last hidden state is considered 
    as sentence representation (see Section 3.2.1 of the paper)
    """
    def __init__(self, hidden_dim, embed_size = 300):
        super(UniLSTM, self).__init__()
        
        self.lstm = torch.nn.LSTM(input_size = embed_size, hidden_size = hidden_dim, 
                                  num_layers = 1, dropout=0, bidirectional=False, 
                                  batch_first=True)

    def forward(self, embedding, length):
        """
        embedding: (batch_size, max_length = 100, embedding_size (GloVe) = 300)
        length: (batch_size)
        """

        x = torch.nn.utils.rnn.pack_padded_sequence(embedding, length.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(x) # hn: (num_layers * num_directions = 1, batch = 64, hidden_dim = 2048)
        emb = hn.squeeze(0) # (batch_size, hidden_dim = 2048)

        return emb

class BiLSTM(torch.nn.Module):
    """
    Simple bidirectional LSTM (BiLSTM), where the last hidden state of forward and backward
    layers are concatenated as the sentence representations
    """
    def __init__(self, hidden_dim, embed_size = 300):
        super(BiLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = embed_size, hidden_size = hidden_dim, 
                                  num_layers = 1, dropout=0, bidirectional=True, 
                                  batch_first=True)
        
    def forward(self, embedding, length):
        """
        embedding: (batch_size, max_length = 100, embedding_size (GloVe) = 300)
        length: (batch_size)
        """
        x = torch.nn.utils.rnn.pack_padded_sequence(embedding, length.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(x)
        emb = torch.cat((hn[0], hn[1]), dim=1) # (batch_size, hidden_dim * num_directions = 4096)

        return emb
    
class MaxPoolLSTM(torch.nn.Module):
    """
    BiLSTM with max pooling applied to the concatenation of word-level hidden states from
    both directions to retrieve sentence representations
    """
    def __init__(self, hidden_dim, embed_size = 300):
        super(MaxPoolLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = embed_size, hidden_size = hidden_dim, 
                                  num_layers = 1, dropout=0, bidirectional=True, 
                                  batch_first=True)
        self.total_length = 100
        
    def forward(self, embedding, length):
        """
        embedding: (batch_size, max_length = 100, embedding_size (GloVe) = 300)
        length: (batch_size)
        """
        x = torch.nn.utils.rnn.pack_padded_sequence(embedding, length.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, (hn, cn) = self.lstm(x)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=self.total_length) # (batch_size, max_length = 100, hidden_dim * num_directions = 4096)

        # Remove zero padding and apply max pooling
        lstm_out = [x[:l] for x, l in zip(lstm_out, length)]
        emb = [torch.max(x, dim = 0).values for x in lstm_out]
        emb = torch.stack(emb, 0)

        return emb



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim = 512, output_dim = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    

class NLINet(nn.Module):
    def __init__(self, encoder, classifier, vocab):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.vocab = vocab

        wordvec = torch.stack(list(vocab.wordvec.values()))
        vocab_size, embed_size = wordvec.shape


        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=1)

        with torch.no_grad():
            self.embedding.weight.copy_(torch.stack(list(vocab.wordvec.values())))
            self.embedding.weight.requires_grad = False

    def concat_sentreps(self, sentrep1, sentrep2):
        return torch.cat([sentrep1, sentrep2, torch.abs(sentrep1 - sentrep2), sentrep1 * sentrep2], dim=1)
    
    # def get_ids(self, sentence):
    #     unk_id = list(self.vocab.wordvec.keys()).index('<unk>')
    #     pad_id = list(self.vocab.wordvec.keys()).index('<pad>')
    #     slen = len(sentence)

    #     print('slen', slen)

    #     # put slen as torch integer


    #     assert slen <= self.maxlen, f"Sentence length exceeds the maximum length of {self.maxlen}"
    #     assert slen > 0, "Sentence length is 0"
    #     assert unk_id == 0, "Unknown token id is not 0"
    #     assert pad_id == 1, "Padding token id is not 1"

    #     indices = [self.vocab.word2id.get(word, unk_id) for word in sentence] + [pad_id] * (self.maxlen - slen)
    #     indices = torch.tensor(indices, dtype=torch.long)
    #     # indices = indices.unsqueeze(0)
    #     print('indices device', indices.device)
    #     return indices, slen
    
    def encode(self, sent_id, slen):
        wordvec = self.embedding(sent_id)
        sentrep = self.encoder(wordvec, slen)
        return sentrep

    def forward(self, x):
        sid1, sid2, len1, len2 = x

        u, v = self.encode(sid1, len1), self.encode(sid2, len2) # (batch_size, embedding_size)
        features = self.concat_sentreps(u, v) # (batch_size, 4 * embedding_size)
        y_hat = self.classifier(features) # (batch_size, 3)

        return y_hat
