class LSTMEncoderAdapter(BaseEncoder):
    def __init__(self, lstm: nn.LSTM, init_hidden: Tuple[Tensor, Tensor]):
        super(LSTMEncoderAdapter, self).__init__()
        self.lstm = lstm
        self.in_channels = self.lstm.input_size
        self.out_channels = self.lstm.hidden_size
        self.init_hidden = init_hidden

    def forward(self, src, src_mask=None, src_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        '''
        Shapes:
        -------
        - src of shape (B, S, E):
        - src_mask of shape (B, S):
        - src_padding_mask of shape (B, S):

        Outputs:
        --------
        - output of shape (B, S, hidden_size * num_directions)
        '''
        src = self.lstm(src, self.init_hidden)[0]
        return src


class GRUEncoderAdapter(BaseEncoder):
    def __init__(self, gru: nn.GRU):
        super(GRUEncoderAdapter, self).__init__()
        self.gru = gru
        self.in_channels = self.gru.input_size
        self.out_channels = self.gru.hidden_size

    def forward(self, src: Tensor) -> Tensor:
        src = self.gru(src)[0]
        return src
