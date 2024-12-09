import torch
import torch.nn as nn

import math
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from utils.models.Embed import TokenEmbedding, PositionalEmbedding, XPOS

class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) based model for sequence classification tasks.

    Attributes:
    -----------
    hidden_size : int
        The number of features in the hidden state `h` of the LSTM.
    num_layers : int
        Number of recurrent layers in the LSTM. E.g., setting `num_layers=2` would mean stacking two LSTMs to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final output.
    lstm : nn.LSTM
        The LSTM layer that performs the sequence learning. It applies dropout between LSTM layers for regularization.
    fc : nn.Linear
        The fully connected layer that maps the output from the last hidden state to the desired number of output classes.

    Methods:
    --------
    forward(x):
        Performs a forward pass of the LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_length, input_size)` where:
            - `batch_size`: Number of samples in a batch.
            - `seq_length`: Number of time steps in the sequence.
            - `input_size`: Dimensionality of the input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        """
        Initializes the LSTM model with the specified parameters.

        Parameters:
        -----------
        input_size : int
            The number of input features at each time step of the sequence.
        hidden_size : int
            The number of features in the hidden state of the LSTM.
        num_layers : int
            The number of stacked LSTM layers.
        num_classes : int
            The number of output classes for the classification task.
        dropout_rate : float, optional (default=0.5)
            Dropout probability applied between LSTM layers to prevent overfitting.
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_length, input_size)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence.
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

class GRUModel(nn.Module):
    """
    A Gated Recurrent Unit (GRU) based model for sequence classification tasks.

    Attributes:
    -----------
    hidden_size : int
        The number of features in the hidden state `h` of the GRU.
    num_layers : int
        Number of recurrent layers in the GRU. E.g., setting `num_layers=2` would mean stacking two GRUs to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final output.
    gru : nn.GRU
        The GRU layer that performs the sequence learning. It applies dropout between GRU layers for regularization.
    fc : nn.Linear
        The fully connected layer that maps the output from the last hidden state to the desired number of output classes.

    Methods:
    --------
    forward(x):
        Performs a forward pass of the GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_length, input_size)` where:
            - `batch_size`: Number of samples in a batch.
            - `seq_length`: Number of time steps in the sequence.
            - `input_size`: Dimensionality of the input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        """
        Initializes the GRU model with the specified parameters.

        Parameters:
        -----------
        input_size : int
            The number of input features at each time step of the sequence.
        hidden_size : int
            The number of features in the hidden state of the GRU.
        num_layers : int
            The number of stacked GRU layers.
        num_classes : int
            The number of output classes for the classification task.
        dropout_rate : float, optional (default=0.5)
            Dropout probability applied between GRU layers to prevent overfitting.
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer with dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, seq_length, input_size)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence.
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

class BasicBlock1d(nn.Module):
    """
    A 1D version of the basic residual block used in ResNet architectures for 1D convolutional operations.

    This block consists of two 1D convolutional layers, each followed by batch normalization and ReLU activation. 
    It also supports a downsampling layer for handling input-output size mismatches. The residual connection
    adds the input directly to the output before the final ReLU activation, helping to improve gradient flow
    during backpropagation.

    Attributes:
    -----------
    expansion : int
        Expansion factor for output channels, which is set to 1 in this basic block.
    conv1 : nn.Conv1d
        The first 1D convolutional layer with kernel size 7, followed by batch normalization and ReLU activation.
    bn1 : nn.BatchNorm1d
        Batch normalization applied after the first convolutional layer.
    relu : nn.ReLU
        ReLU activation applied after each batch normalization layer.
    dropout : nn.Dropout
        Dropout layer with a dropout probability of 0.2, applied after the first ReLU activation for regularization.
    conv2 : nn.Conv1d
        The second 1D convolutional layer with kernel size 7, followed by batch normalization and ReLU activation.
    bn2 : nn.BatchNorm1d
        Batch normalization applied after the second convolutional layer.
    downsample : Optional[nn.Module]
        A downsampling layer (if specified) that adjusts the size of the residual input to match the output size.

    Methods:
    --------
    forward(x):
        Performs a forward pass of the BasicBlock1d module.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, in_channels, length)`, where:
            - `batch_size`: Number of samples in a batch.
            - `in_channels`: Number of input channels.
            - `length`: Length of the input sequence.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, out_channels, output_length)`, where:
            - `out_channels`: Number of output channels (same as `planes` argument passed during initialization).
            - `output_length`: Length of the output sequence, which may differ depending on the stride and downsampling.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes the BasicBlock1d with two convolutional layers and optional downsampling.

        Parameters:
        -----------
        inplanes : int
            Number of input channels for the first convolutional layer.
        planes : int
            Number of output channels for both convolutional layers.
        stride : int, optional (default=1)
            Stride for the first convolutional layer. Controls the downsampling of the input.
        downsample : Optional[nn.Module], optional
            A module to downsample the input in the residual connection, matching input size to output size.
        """
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        """
        Forward pass through the BasicBlock1d.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, in_channels, length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, planes, output_length)` after residual addition and ReLU activation.
        """
        residual = x
        
        # First convolutional layer, batch normalization, ReLU, and dropout
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolutional layer and batch normalization
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsampling the input if required
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Adding the residual connection
        out += residual
        
        # Final ReLU activation
        out = self.relu(out)
        
        return out

class ResNet1d(nn.Module):
    """
    A 1D version of the ResNet architecture for time-series or sequential data classification tasks.

    This model builds a 1D residual network with multiple layers of convolutional blocks, followed by both adaptive average pooling and adaptive max pooling layers to capture relevant features from sequential data. It uses a fully connected layer at the end to perform classification.

    Attributes:
    -----------
    inplanes : int
        Number of input channels for the first convolutional layer.
    conv1 : nn.Conv1d
        The first 1D convolutional layer that reduces the input dimensionality and prepares the input for subsequent layers.
    bn1 : nn.BatchNorm1d
        Batch normalization layer applied after the first convolution.
    relu : nn.ReLU
        ReLU activation function applied after batch normalization.
    maxpool : nn.MaxPool1d
        Max pooling layer to downsample the input after the first block.
    layer1, layer2, layer3, layer4 : nn.Sequential
        Four residual layers, each consisting of multiple residual blocks created using `_make_layer`. These layers progressively increase the feature map size and apply residual connections to improve gradient flow.
    adaptiveavgpool : nn.AdaptiveAvgPool1d
        Adaptive average pooling layer that reduces the spatial dimensions of the output to a fixed size (1).
    adaptivemaxpool : nn.AdaptiveMaxPool1d
        Adaptive max pooling layer that also reduces the spatial dimensions of the output to a fixed size (1).
    fc : nn.Linear
        Fully connected layer that maps the pooled output to the number of target classes for classification.
    dropout : nn.Dropout
        Dropout layer with a probability of 0.2, applied before the fully connected layer to reduce overfitting.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer with a given number of blocks and planes. Each layer can have a downsampling step if needed.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to be used (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the convolutional layers in the block.
        blocks : int
            The number of residual blocks in this layer.
        stride : int, optional (default=1)
            The stride to be applied to the convolutional layers. Controls downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    forward(x):
        Performs a forward pass of the ResNet1d model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9):
        """
        Initializes the ResNet1d model with a given block structure and layer configuration.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use in the model (e.g., BasicBlock1d).
        layers : list of int
            List containing the number of blocks in each of the four residual layers.
        input_channels : int, optional (default=12)
            The number of input channels in the data (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of channels to use in the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        """
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        
        # Initial convolutional layer followed by batch normalization, ReLU, and max pooling
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Four residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Adaptive pooling and fully connected layers
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNet1d model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Passing through residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Adaptive pooling
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        
        # Concatenate outputs from adaptive average and max pooling
        x = torch.cat((x1, x2), dim=1)
        
        # Flatten and apply dropout
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layer for final output
        return self.fc(x)

class Residual_Conv_Mamba(nn.Module):
    """
    A hybrid model combining ResNet-inspired convolutional layers with Mamba residual blocks for sequence data classification.

    This model integrates convolutional layers from a ResNet-like structure with Mamba residual blocks. It is designed for sequential or time-series data classification tasks, using a combination of convolutional feature extraction and transformer-based residual learning layers.

    Attributes:
    -----------
    inplanes : int
        Number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers (inspired by ResNet), which downsample the input data and extract high-level features.
    embedding : nn.Linear
        A linear layer that projects the ResNet features into a higher-dimensional space (of size `d_model`) suitable for input into the Mamba residual blocks.
    d_inner : int
        The expanded dimension of the feed-forward layer in the Mamba residual block, calculated as `d_model * expand`.
    dt_rank : int
        The rank for the dynamic tensor approximation within the Mamba residual block, calculated as `ceil(d_model / 16)`.
    Mambalayers : nn.ModuleList
        A list of Mamba residual blocks for transformer-based residual learning. The number of layers is specified by `e_layers`.
    norm : RMSNorm
        Root Mean Square Layer Normalization applied to stabilize the output after the Mamba residual blocks.
    fc : nn.Linear
        Fully connected layer that outputs the predicted class scores.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer using the specified block and number of planes (output channels), along with an optional stride for downsampling.

        Parameters:
        -----------
        block : nn.Module
            The type of block to use for building the residual layer (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the convolutional layers in the block.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    forward(x):
        Performs a forward pass of the Residual_Conv_Mamba model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
    """

    def __init__(self, d_model, d_conv, d_ff, expand, e_layers=2, input_channels=12, inplanes=64, num_classes=9):
        """
        Initializes the Residual_Conv_Mamba model with convolutional and Mamba residual block layers.

        Parameters:
        -----------
        d_model : int
            The dimensionality of the feature space output by the embedding layer and input to the Mamba residual blocks.
        d_conv : int
            The dimensionality of the convolutional layers in the Mamba residual block.
        d_ff : int
            The dimensionality of the feed-forward network within the Mamba residual block.
        expand : int
            Expansion factor for the feed-forward network's hidden size.
        e_layers : int, optional (default=2)
            The number of Mamba residual blocks to stack.
        input_channels : int, optional (default=12)
            The number of input channels in the data (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of channels to use in the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        """
        super(Residual_Conv_Mamba, self).__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )
        
        # Embedding layer to project the ResNet output to a higher-dimensional space
        self.embedding = nn.Linear(1875, d_model)

        # Mamba residual block configuration
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        self.Mambalayers = nn.ModuleList([ResidualBlock(d_model, d_conv, d_ff, self.d_inner, self.dt_rank) for _ in range(e_layers)])
        
        # Normalization and fully connected layers
        self.norm = RMSNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the Residual_Conv_Mamba model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence.
        """
        # ResNet-like convolutional feature extraction
        x = self.resnet_half(x)
        
        # Projecting the convolutional output into the d_model space
        x = self.embedding(x)
        
        # Passing through Mamba residual blocks
        for layer in self.Mambalayers:
            x = layer(x)
        
        # Normalization
        x = self.norm(x)
        
        # Fully connected layer for classification
        x = self.fc(x.mean(dim=1))
        return x

class Residual_Conv_GRU(nn.Module):
    """
    A hybrid model combining ResNet-inspired convolutional layers with a Gated Recurrent Unit (GRU) for sequence data classification.

    This model extracts high-level features using ResNet-like convolutional layers and processes sequential dependencies using GRU layers. It is designed for time-series data or sequence classification tasks.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers inspired by ResNet that downsample the input and extract features.
    gru : nn.GRU
        A Gated Recurrent Unit (GRU) network with multiple layers for sequence modeling. It processes the output of the convolutional layers.
    fc : nn.Linear
        Fully connected layer that maps the output of the GRU to the target number of classes.
    num_layers : int
        Number of GRU layers.
    hidden_size : int
        The number of features in the hidden state of the GRU.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    get_output_shape(input_size, batch_size, input_channels):
        Calculates the output shape after passing data through the ResNet layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The number of samples in a batch.
        input_channels : int
            The number of input channels (e.g., number of features in the data).

        Returns:
        --------
        int
            The size of the feature map after convolutional processing.

    forward(x, quantize=None):
        Performs a forward pass of the Residual_Conv_GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        quantize : bool, optional (default=None)
            If set to True, quantizes the hidden state to `float16` for memory efficiency.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, GRU_hidden_size=128, GRU_num_layers=2):
        """
        Initializes the Residual_Conv_GRU model with convolutional and GRU layers.

        Parameters:
        -----------
        input_size : int
            The size of the input features (length of the sequence).
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        GRU_hidden_size : int, optional (default=128)
            The number of features in the hidden state of the GRU.
        GRU_num_layers : int, optional (default=2)
            The number of layers in the GRU.
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )

        # GRU layer for sequence processing
        self.gru = nn.GRU(input_size=128, hidden_size=GRU_hidden_size, num_layers=GRU_num_layers, batch_first=True, dropout=0.1)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(GRU_hidden_size, num_classes)
        
        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_output_shape(self, input_size, batch_size, input_channels):
        """
        Computes the output shape after the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size of the input data.
        input_channels : int
            The number of input channels (e.g., features in the input data).

        Returns:
        --------
        int
            The size of the feature map after processing through the ResNet layers.
        """
        dim = (batch_size, input_channels, input_size)
        return self.resnet_half(torch.zeros(dim)).size(2)

    def forward(self, x, quantize=None):
        """
        Forward pass through the Residual_Conv_GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        quantize : bool, optional (default=None)
            If True, initializes the GRU hidden state in `float16` for memory efficiency.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)
        
        # Rearrange the dimensions for GRU input
        x = x.permute(0, 2, 1)
        
        # Initialize the GRU hidden state
        if quantize:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=torch.float16).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through the GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class LoRALayer(nn.Module):
    def __init__(self, base_layer, r, alpha, dropout=0.0):
        super(LoRALayer, self).__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if isinstance(base_layer, nn.Conv1d):
            # Conv1d 的情況
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0]

            # LoRA 的權重設置
            self.lora_A = nn.Parameter(torch.zeros(r, in_channels * kernel_size))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
        else:
            # Linear 的情況
            self.lora_A = nn.Parameter(torch.zeros(r, base_layer.weight.shape[1]))
            self.lora_B = nn.Parameter(torch.zeros(base_layer.weight.shape[0], r))

        # 初始化 LoRA 權重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base_layer(x)

        if isinstance(self.base_layer, nn.Conv1d):
            # Conv1d 的情況：展開輸入
            batch_size, in_channels, seq_len = x.size()
            kernel_size = self.base_layer.kernel_size[0]
            stride = self.base_layer.stride[0]
            padding = self.base_layer.padding[0]

            # 使用 unfold 展開輸入，形狀變為 (batch_size, 展開窗口大小, 窗口數量)
            x_unfold = F.unfold(
                x.unsqueeze(-1), 
                kernel_size=(kernel_size, 1), 
                stride=(stride, 1), 
                padding=(padding, 0)
            ).view(batch_size, in_channels * kernel_size, -1)

            # LoRA 適配
            lora_out = self.lora_A @ x_unfold
            lora_out = self.lora_B @ lora_out
            lora_out = lora_out.view(batch_size, self.base_layer.out_channels, -1)

            return base_out + self.scaling * lora_out
        else:
            # Linear 的情況
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            return base_out + self.scaling * lora_out

class Residual_Conv_GRU_LoRA(nn.Module):
    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9,
                 GRU_hidden_size=128, GRU_num_layers=2, lora_r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction with LoRA
        self.resnet_half = nn.Sequential(
            LoRALayer(nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
                      r=16, alpha=64, dropout=0.1),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )

        # GRU layer for sequence processing
        self.gru = nn.GRU(input_size=128, hidden_size=GRU_hidden_size,
                          num_layers=GRU_num_layers, batch_first=True, dropout=0.1)
        # Apply LoRA to the input of GRU
        self.gru_lora = LoRALayer(nn.Linear(128, GRU_hidden_size),
                                  r=16, alpha=32, dropout=0.2)

        # Fully connected layer with LoRA
        self.fc = LoRALayer(nn.Linear(GRU_hidden_size, num_classes),
                            r=16, alpha=32, dropout=0.0)

        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                LoRALayer(nn.Conv1d(self.inplanes, planes * block.expansion,
                                    kernel_size=1, stride=stride, bias=False),
                          r=4, alpha=16, dropout=0.1),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, quantize=None):
        # Apply ResNet layers
        x = self.resnet_half(x)
        x = x.permute(0, 2, 1)  # Rearrange for GRU input

        # Initialize GRU hidden state
        if quantize:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                             dtype=torch.float16).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Apply LoRA adaptation to GRU input
        x = self.gru_lora(x)
        out, _ = self.gru(x, h0)  # Forward propagate through GRU
        out = self.fc(out[:, -1, :])  # Decode the last time step
        return out


class Residual_Conv_LSTM(nn.Module):
    """
    A hybrid model combining ResNet-inspired convolutional layers with an LSTM network for sequence data classification.

    This model extracts features using ResNet-like convolutional layers and processes sequential dependencies using LSTM layers. It is designed for time-series data or sequence classification tasks, such as ECG classification or other temporal data.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers (inspired by ResNet) that downsample the input data and extract high-level features.
    lstm : nn.LSTM
        A Long Short-Term Memory (LSTM) network with multiple layers for sequence modeling. It processes the output of the convolutional layers.
    fc : nn.Linear
        Fully connected layer that maps the output of the LSTM to the target number of classes.
    num_layers : int
        The number of LSTM layers.
    hidden_size : int
        The number of features in the hidden state of the LSTM.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    get_output_shape(input_size, batch_size, input_channels):
        Computes the output shape after passing data through the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The number of samples in a batch.
        input_channels : int
            The number of input channels (e.g., features in the data).

        Returns:
        --------
        int
            The size of the feature map after convolutional processing.

    forward(x):
        Performs a forward pass of the Residual_Conv_LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """

    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, hidden_size=128, num_layers=2):
        """
        Initializes the Residual_Conv_LSTM model with convolutional and LSTM layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        hidden_size : int, optional (default=128)
            The number of features in the hidden state of the LSTM.
        num_layers : int, optional (default=2)
            The number of layers in the LSTM.
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(self.get_output_shape(input_size, batch_size, input_channels), hidden_size, num_layers, batch_first=True, dropout=0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_output_shape(self, input_size, batch_size, input_channels):
        """
        Computes the output shape after the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size of the input data.
        input_channels : int
            The number of input channels (e.g., features in the input data).

        Returns:
        --------
        int
            The size of the feature map after processing through the ResNet layers.
        """
        dim = (batch_size, input_channels, input_size)
        return self.resnet_half(torch.zeros(dim)).size(2)

    def forward(self, x):
        """
        Forward pass through the Residual_Conv_LSTM model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)

        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through the LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out
    
class mini_Residual_Conv_GRU(nn.Module):
    """
    A smaller version of the Residual_Conv_GRU model, combining ResNet-inspired convolutional layers with a Gated Recurrent Unit (GRU) for sequence data classification.

    This model first extracts features using ResNet-like convolutional layers, followed by a GRU to capture temporal dependencies in the data. It is designed for time-series or sequential data classification tasks, such as ECG classification or other temporal data.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers inspired by ResNet that downsample the input data and extract features.
    gru : nn.GRU
        A Gated Recurrent Unit (GRU) network with multiple layers for sequence modeling. It processes the output of the convolutional layers.
    fc : nn.Linear
        Fully connected layer that maps the output of the GRU to the target number of classes.
    num_layers : int
        The number of GRU layers.
    hidden_size : int
        The number of features in the hidden state of the GRU.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    get_output_shape(input_size, batch_size, input_channels):
        Computes the output shape after passing data through the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The number of samples in a batch.
        input_channels : int
            The number of input channels (e.g., features in the data).

        Returns:
        --------
        int
            The size of the feature map after convolutional processing.

    forward(x):
        Performs a forward pass of the mini_Residual_Conv_GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, GRU_hidden_size=128, GRU_num_layers=2):
        """
        Initializes the mini_Residual_Conv_GRU model with convolutional and GRU layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        GRU_hidden_size : int, optional (default=128)
            The number of features in the hidden state of the GRU.
        GRU_num_layers : int, optional (default=2)
            The number of layers in the GRU.
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # GRU layer for sequence modeling
        self.gru = nn.GRU(input_size=self.get_output_shape(input_size, batch_size, input_channels),
                          hidden_size=GRU_hidden_size, num_layers=GRU_num_layers, batch_first=True, dropout=0.5)

        # Fully connected layer for classification
        self.fc = nn.Linear(GRU_hidden_size, num_classes)

        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_output_shape(self, input_size, batch_size, input_channels):
        """
        Computes the output shape after the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size of the input data.
        input_channels : int
            The number of input channels (e.g., features in the input data).

        Returns:
        --------
        int
            The size of the feature map after processing through the ResNet layers.
        """
        dim = (batch_size, input_channels, input_size)
        return self.resnet_half(torch.zeros(dim)).size(2)

    def forward(self, x):
        """
        Forward pass through the mini_Residual_Conv_GRU model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)

        # Initialize the GRU hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through the GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

class Transformer(nn.Module):
    """
    A Transformer-based model for sequence data classification.

    This model uses a 1D convolutional layer for initial feature projection followed by positional encoding and Transformer encoder layers. The final classification is performed using a fully connected layer. The model is suitable for time-series or sequence classification tasks.

    Attributes:
    -----------
    in_proj : nn.Conv1d
        A 1D convolutional layer that projects the input features to the model dimension (`model_dim`).
    pos_encoder : PositionalEncoding
        A module that adds positional encoding to the input sequences to provide the model with temporal context.
    transformer_encoder : nn.TransformerEncoder
        A stack of Transformer encoder layers that process the input sequences.
    fc : nn.Linear
        A fully connected layer that maps the Transformer output to the number of target classes.
    model_dim : int
        The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).

    Methods:
    --------
    forward(x, src_mask=None):
        Performs a forward pass of the Transformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, num_classes, num_layers, model_dim=64):
        """
        Initializes the Transformer model with convolutional projection, positional encoding, Transformer layers, and a fully connected output layer.

        Parameters:
        -----------
        num_classes : int
            The number of output classes for classification.
        num_layers : int
            The number of Transformer encoder layers to stack.
        model_dim : int, optional (default=64)
            The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).
        """
        super(Transformer, self).__init__()
        
        # 1D convolutional layer for input feature projection
        self.in_proj = nn.Conv1d(12, model_dim, kernel_size=30, stride=2, padding=7, bias=False)
        
        # Positional encoding to add temporal context
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=4, dim_feedforward=512, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(model_dim, num_classes)
        
        self.model_dim = model_dim

    def forward(self, x, src_mask=None):
        """
        Forward pass through the Transformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Initial projection with convolutional layer
        x = self.in_proj(x)
        
        # Permute dimensions for Transformer input (batch_size, sequence_length, model_dim)
        x = x.permute(0, 2, 1)
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Pass through Transformer encoder layers
        x = self.transformer_encoder(x, src_mask)
        
        # Average over the sequence length dimension and apply fully connected layer
        output = self.fc(x.mean(dim=1))
        
        return output

class Residual_ConvTransformer(nn.Module):
    """
    A hybrid model that combines ResNet-inspired convolutional layers with a Transformer encoder for sequence data classification.

    This model uses convolutional layers to extract high-level features from sequential data, followed by Transformer layers to model temporal dependencies. The final classification is performed using a fully connected layer.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of ResNet-inspired convolutional layers that downsample the input and extract features.
    embedding : nn.Linear
        A linear layer that projects the ResNet output to the Transformer model dimension (`model_dim`).
    pos_encoder : PositionalEncoding
        A module that adds positional encoding to the input sequences, providing the Transformer with temporal context.
    transformer_encoder : nn.TransformerEncoder
        A stack of Transformer encoder layers that process the input sequences.
    fc : nn.Linear
        A fully connected layer that maps the Transformer output to the target number of classes.
    model_dim : int
        The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    forward(x, src_mask=None):
        Performs a forward pass of the Residual_ConvTransformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, num_layers=2, model_dim=64):
        """
        Initializes the Residual_ConvTransformer model with convolutional and Transformer layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        num_layers : int, optional (default=2)
            The number of Transformer encoder layers to stack.
        model_dim : int, optional (default=64)
            The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )

        # Linear embedding layer for Transformer input
        self.embedding = nn.Linear(128, model_dim)

        # Positional encoding to provide temporal context
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=32, dim_feedforward=512, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(model_dim, num_classes)

        self.model_dim = model_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, src_mask=None):
        """
        Forward pass through the Residual_ConvTransformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)

        # Permute for Transformer input (batch_size, seq_length, model_dim)
        x = x.permute(0, 2, 1)

        # Project to Transformer model dimension and scale
        x = self.embedding(x) * math.sqrt(self.model_dim)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Pass through Transformer encoder layers
        x = self.transformer_encoder(x, src_mask)

        # Average over the sequence length and apply fully connected layer
        output = self.fc(x.mean(dim=1))

        return output
    
class mini_Residual_ConvTransformer(nn.Module):
    """
    A smaller version of the Residual_ConvTransformer model that combines ResNet-inspired convolutional layers with a Transformer encoder for sequence data classification.

    This model first extracts features using ResNet-like convolutional layers and then processes sequential dependencies using Transformer layers. The final classification is performed using a fully connected layer. This smaller version is designed for efficiency while maintaining a strong ability to model temporal data.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers that downsample the input data and extract high-level features.
    embedding : nn.Linear
        A linear layer that projects the output of the convolutional layers to the Transformer model dimension (`model_dim`).
    pos_encoder : PositionalEncoding
        A module that adds positional encoding to the input sequences to provide temporal context for the Transformer.
    transformer_encoder : nn.TransformerEncoder
        A stack of Transformer encoder layers that process the input sequences.
    fc : nn.Linear
        A fully connected layer that maps the Transformer output to the target number of classes.
    model_dim : int
        The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).

    Methods:
    --------
    get_output_shape(input_size, batch_size, input_channels):
        Computes the output shape after passing data through the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The number of samples in a batch.
        input_channels : int
            The number of input channels (e.g., features in the data).

        Returns:
        --------
        int
            The size of the feature map after convolutional processing.

    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    forward(x, src_mask=None):
        Performs a forward pass of the mini_Residual_ConvTransformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """
    
    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, num_layers=2, model_dim=64):
        """
        Initializes the mini_Residual_ConvTransformer model with convolutional and Transformer layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        num_layers : int, optional (default=2)
            The number of Transformer encoder layers to stack.
        model_dim : int, optional (default=64)
            The dimensionality of the model (i.e., the size of the input/output features for each sequence in the Transformer).
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Linear embedding layer for Transformer input
        self.embedding = nn.Linear(64, model_dim)

        # Positional encoding to provide temporal context
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Fully connected layer for classification
        self.fc = nn.Linear(model_dim, num_classes)

        self.model_dim = model_dim

    def get_output_shape(self, input_size, batch_size, input_channels):
        """
        Computes the output shape after the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size of the input data.
        input_channels : int
            The number of input channels (e.g., features in the input data).

        Returns:
        --------
        int
            The size of the feature map after processing through the ResNet layers.
        """
        dim = (batch_size, input_channels, input_size)
        return self.resnet_half(torch.zeros(dim)).size(2)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, src_mask=None):
        """
        Forward pass through the mini_Residual_ConvTransformer model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.
        src_mask : torch.Tensor, optional (default=None)
            Source mask for the Transformer encoder, used to apply attention masking if needed.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)

        # Permute for Transformer input (batch_size, seq_length, model_dim)
        x = x.permute(0, 2, 1)

        # Project to Transformer model dimension and scale
        x = self.embedding(x) * math.sqrt(self.model_dim)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Pass through Transformer encoder layers
        x = self.transformer_encoder(x, src_mask)

        # Average over the sequence length and apply fully connected layer
        output = self.fc(x.mean(dim=1))

        return output
    
class Residual_conv_retnet(nn.Module):
    """
    A hybrid model that combines ResNet-inspired convolutional layers with RetNet (Retention Network) layers for sequence data classification.

    This model extracts features using ResNet-like convolutional layers and processes sequential dependencies using multi-scale retention layers (RetNet). The final classification is performed using a fully connected layer. The model is designed for time-series or sequence classification tasks.

    Attributes:
    -----------
    inplanes : int
        The number of input channels for the first convolutional layer in the ResNet block.
    resnet_half : nn.Sequential
        A stack of convolutional layers inspired by ResNet that downsample the input data and extract high-level features.
    layers : int
        The number of RetNet layers (multi-scale retention layers) to stack.
    hidden_dim : int
        The dimensionality of the hidden representation in the RetNet layers.
    ffn_size : int
        The size of the feed-forward network in each RetNet layer.
    heads : int
        The number of attention heads in the multi-scale retention layers.
    double_v_dim : bool
        Flag indicating whether to double the dimension of the retention vector (`v_dim`).
    v_dim : int
        The dimensionality of the retention vector, calculated based on `hidden_dim` and `double_v_dim`.
    embedding : nn.Linear
        A linear layer that projects the ResNet output into the `hidden_dim` space for the RetNet layers.
    pos_encoder : PositionalEncoding
        A module that adds positional encoding to the input sequences, providing temporal context for the RetNet layers.
    retentions : nn.ModuleList
        A list of multi-scale retention layers (RetNet) for processing sequential dependencies.
    ffns : nn.ModuleList
        A list of feed-forward networks, one for each RetNet layer.
    layer_norms_1 : nn.ModuleList
        A list of layer normalization modules, one for each RetNet layer, applied before the retention layer.
    fc : nn.Linear
        A fully connected layer that maps the final hidden representation to the number of target classes.

    Methods:
    --------
    _make_layer(block, planes, blocks, stride=1):
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.

    get_output_shape(input_size, batch_size, input_channels):
        Computes the output shape after passing data through the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The number of samples in a batch.
        input_channels : int
            The number of input channels (e.g., features in the data).

        Returns:
        --------
        int
            The size of the feature map after convolutional processing.

    forward(x):
        Performs a forward pass of the Residual_conv_retnet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sample in the batch.
    """

    def __init__(self, input_size, batch_size, input_channels=12, inplanes=64, num_classes=9, num_layers=2, hidden_dim=128, ffn_size=128, quantize=False):
        """
        Initializes the Residual_conv_retnet model with convolutional and RetNet layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size used for input.
        input_channels : int, optional (default=12)
            The number of input channels (e.g., number of channels in an ECG).
        inplanes : int, optional (default=64)
            The number of input channels to use for the first convolutional layer.
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        num_layers : int, optional (default=2)
            The number of RetNet layers to stack.
        hidden_dim : int, optional (default=128)
            The dimensionality of the hidden representation in the RetNet layers.
        ffn_size : int, optional (default=128)
            The size of the feed-forward network in each RetNet layer.
        quantize : bool, optional (default=False)
            Flag indicating whether quantization is applied to the multi-scale retention layers.
        """
        super().__init__()
        self.inplanes = inplanes

        # ResNet-like convolutional feature extraction
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
        )

        # RetNet layers
        self.layers = num_layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = 8
        self.double_v_dim = False
        self.v_dim = hidden_dim * 2 if self.double_v_dim else hidden_dim

        # Linear embedding layer
        self.embedding = nn.Linear(self.get_output_shape(input_size, batch_size, input_channels), hidden_dim)

        # Positional encoding to provide temporal context
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)

        # Multi-scale retention layers
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, self.heads, self.double_v_dim, quantize)
            for _ in range(num_layers)
        ])

        # Feed-forward layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Constructs a residual layer by stacking multiple blocks.

        Parameters:
        -----------
        block : nn.Module
            The type of residual block to use (e.g., BasicBlock1d).
        planes : int
            The number of output channels for the blocks.
        blocks : int
            The number of residual blocks to stack in this layer.
        stride : int, optional (default=1)
            The stride applied to the first convolutional layer in the block, allowing for downsampling.

        Returns:
        --------
        nn.Sequential
            A sequential container of residual blocks forming the layer.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_output_shape(self, input_size, batch_size, input_channels):
        """
        Computes the output shape after the ResNet convolutional layers.

        Parameters:
        -----------
        input_size : int
            The length of the input sequence.
        batch_size : int
            The batch size of the input data.
        input_channels : int
            The number of input channels (e.g., features in the input data).

        Returns:
        --------
        int
            The size of the feature map after processing through the ResNet layers.
        """
        dim = (batch_size, input_channels, input_size)
        return self.resnet_half(torch.zeros(dim)).size(2)

    def forward(self, x):
        """
        Forward pass through the Residual_conv_retnet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, input_channels, sequence_length)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply ResNet convolutional layers
        x = self.resnet_half(x)

        # Linear projection for RetNet input
        x = self.embedding(x) * math.sqrt(self.hidden_dim)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Forward pass through RetNet layers
        for i in range(self.layers):
            x = self.retentions[i](self.layer_norms_1[i](x)) + x

        # Fully connected layer for classification
        x = self.fc(x.mean(dim=1))

        return x

class PositionalEncoding(nn.Module):
    """
    A module that adds positional encoding to the input sequences, providing temporal context for models like Transformers.

    Positional encoding is necessary because models like Transformers do not inherently have any information about the order of input tokens. This module encodes positional information using sine and cosine functions of varying frequencies and adds it to the input embeddings.

    Attributes:
    -----------
    dropout : nn.Dropout
        A dropout layer applied to the output of the positional encoding for regularization.
    pe : torch.Tensor
        A tensor that contains the positional encodings, precomputed for all positions up to `max_len` and the specified `model_dim`.

    Methods:
    --------
    forward(x):
        Adds positional encoding to the input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(sequence_length, batch_size, model_dim)` representing the embeddings of a sequence.

        Returns:
        --------
        torch.Tensor
            The input tensor with positional encodings added, of the same shape as the input tensor.
    """
    
    def __init__(self, model_dim, dropout, max_len=5000):
        """
        Initializes the PositionalEncoding module.

        Parameters:
        -----------
        model_dim : int
            The dimensionality of the input embeddings (i.e., the number of features for each position in the sequence).
        dropout : float
            The dropout probability applied to the positional encoding for regularization.
        max_len : int, optional (default=5000)
            The maximum length of input sequences for which positional encodings are precomputed.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, model_dim) that contains the positional encodings
        pe = torch.zeros(max_len, model_dim)

        # Compute the positional encodings based on sine and cosine functions
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        # Add a batch dimension and transpose for easier integration with input embeddings
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register `pe` as a buffer, which means it won't be updated during training but can be saved with the model
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(sequence_length, batch_size, model_dim)` representing the embeddings of a sequence.

        Returns:
        --------
        torch.Tensor
            The input tensor with positional encodings added, of the same shape as the input tensor.
        """
        # Add the positional encoding to the input tensor
        x = x + self.pe[:x.size(0), :]
        
        # Apply dropout to the output for regularization
        return self.dropout(x)

#CLINet
class CLINet(nn.Module):
    """
    A neural network model combining convolutional layers, LSTM layers, and involution layers for sequence data classification.

    The CLINet model consists of multiple 1D convolutional layers followed by batch normalization, an LSTM layer for sequential data processing, and involution layers to capture local patterns. The final classification is performed using fully connected layers. This architecture is suitable for tasks such as ECG classification or other temporal data classification problems.

    Attributes:
    -----------
    num_features : int
        The number of input features (e.g., channels in an ECG).
    sequence_len : int
        The length of the input sequences.
    num_classes : int
        The number of output classes for classification.
    conv1, conv2, conv3 : nn.Conv1d
        Three 1D convolutional layers with varying kernel sizes for initial feature extraction.
    batch_norm1 : nn.BatchNorm1d
        Batch normalization applied after concatenating the convolutional outputs.
    relu : nn.ReLU
        ReLU activation function applied throughout the network.
    lstm : nn.LSTM
        An LSTM layer for capturing sequential dependencies in the data.
    inv1, inv2, inv3 : Involution
        Three involution layers that process the sequence output from the LSTM to capture local patterns.
    batch_norm2 : nn.BatchNorm2d
        Batch normalization applied after the involution layers.
    flatten : nn.Flatten
        A layer that flattens the output before passing it to the fully connected layers.
    dropout1, dropout2 : nn.Dropout
        Dropout layers for regularization applied before the fully connected layers.
    fc1, fc2, fc3 : nn.Linear
        Fully connected layers that map the final representation to the number of output classes.

    Methods:
    --------
    forward(x):
        Performs a forward pass of the CLINet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, num_features, sequence_len)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
    """
    
    def __init__(self, sequence_len, num_features=12, num_classes=9):
        """
        Initializes the CLINet model with convolutional, LSTM, and involution layers.

        Parameters:
        -----------
        sequence_len : int
            The length of the input sequences.
        num_features : int, optional (default=12)
            The number of input features (e.g., channels in an ECG).
        num_classes : int, optional (default=9)
            The number of output classes for classification.
        """
        super(CLINet, self).__init__()
        self.num_features = num_features
        self.sequence_len = sequence_len
        self.num_classes = num_classes

        # Convolutional layers with varying kernel sizes
        self.conv1 = nn.Conv1d(num_features, 3, kernel_size=31, stride=5, padding=15)
        self.conv2 = nn.Conv1d(num_features, 3, kernel_size=36, stride=5, padding=16)
        self.conv3 = nn.Conv1d(num_features, 3, kernel_size=41, stride=5, padding=20)

        # Batch normalization and activation
        self.batch_norm1 = nn.BatchNorm1d(9)
        self.relu = nn.ReLU()

        # LSTM for sequential modeling
        self.lstm = nn.LSTM(input_size=9, hidden_size=200, batch_first=True)

        # Involution layers to capture local patterns
        self.inv1 = Involution(channel=3, group_number=1, kernel_size=31, stride=5, reduction_ratio=2)
        self.inv2 = Involution(channel=3, group_number=1, kernel_size=36, stride=5, reduction_ratio=2)
        self.inv3 = Involution(channel=3, group_number=1, kernel_size=41, stride=5, reduction_ratio=2)

        # Batch normalization after involution layers
        self.batch_norm2 = nn.BatchNorm2d(3)

        # Flatten layer and fully connected layers
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(72000, 20)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, self.num_classes)

    def forward(self, x):
        """
        Forward pass through the CLINet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape `(batch_size, num_features, sequence_len)`.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape `(batch_size, num_classes)` representing the predicted class scores for each sequence in the batch.
        """
        # Apply the convolutional layers
        x11 = self.conv1(x)
        x12 = self.conv2(x)
        x13 = self.conv3(x)
        x = torch.cat((x11, x12, x13), dim=1)

        # Apply batch normalization and ReLU
        x = self.batch_norm1(x)
        x = self.relu(x)

        # Permute for LSTM input
        x = x.permute(0, 2, 1)  # Change to (batch_size, sequence_length, num_features)

        # LSTM layer
        x, _ = self.lstm(x)
        x = x.unsqueeze(1)

        # Apply involution layers
        x21 = self.inv1(x)
        x22 = self.inv2(x)
        x23 = self.inv3(x)
        x = torch.cat((x21, x22, x23), dim=1)

        # Apply batch normalization, flatten, and fully connected layers
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

class Involution(nn.Module):
    """
    The Involution class implements an involution operation, a type of neural network layer designed to reduce 
    computational complexity while retaining the capability to capture local information. It operates by generating 
    dynamic kernels based on the input feature map and applying these kernels to the input through a channel-wise 
    grouping mechanism.

    Attributes:
    -----------
    channel : int
        The number of input and output channels.
    group_number : int
        The number of groups to divide the input channels into. The kernel is applied independently across groups.
    kernel_size : int
        The size of the kernel used in the involution operation (assumed to be square).
    stride : int
        The stride of the involution operation. If stride > 1, an average pooling layer is applied before the operation.
    reduction_ratio : int
        A ratio used to reduce the number of channels when generating the kernel. Smaller values lead to more 
        computationally efficient kernels.
    name : str, optional
        An optional name for the layer, default is None.
    stride_layer : nn.Module
        The layer used to downsample the input when stride > 1. If stride = 1, an identity layer is used.
    kernel_gen : nn.Sequential
        A sequential model used to generate the dynamic kernel. It consists of two convolutional layers, 
        batch normalization, and ReLU activation.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the involution layer.

        Parameters:
        -----------
        x : torch.Tensor
            The input tensor with shape (batch_size, num_channels, height, width).

        Returns:
        --------
        output : torch.Tensor
            The output tensor after applying the involution operation, with shape 
            (batch_size, num_channels, height // stride, width // stride).
        kernel : torch.Tensor
            The generated kernel tensor, with shape 
            (batch_size, height // stride, width // stride, kernel_size * kernel_size, group_number).
    """

    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio, name=None):
        """
        Initializes the Involution layer.

        Parameters:
        -----------
        channel : int
            Number of input/output channels.
        group_number : int
            Number of groups to divide the channels into.
        kernel_size : int
            Size of the dynamic convolution kernel.
        stride : int
            Stride of the involution operation.
        reduction_ratio : int
            Reduction ratio for kernel generation.
        name : str, optional
            Optional name for the layer, default is None.
        """
        super(Involution, self).__init__()
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio
        self.stride_layer = (
            nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
            if self.stride > 1 else nn.Identity()
        )
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(1, channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(channel // reduction_ratio, kernel_size * kernel_size * group_number, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass of the involution operation.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
        --------
        output : torch.Tensor
            Output tensor after the involution operation, of shape (batch_size, num_channels, height // stride, width // stride).
        kernel : torch.Tensor
            Generated kernel for the involution operation, with shape 
            (batch_size, height // stride, width // stride, kernel_size * kernel_size, group_number).
        """
        batch_size, num_channels, ori_height, ori_width = x.size()
        height = ori_height // self.stride
        width = ori_width // self.stride
        
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        
        kernel = kernel.view(batch_size, self.group_number, self.kernel_size * self.kernel_size, height, width)
        kernel = kernel.permute(0, 3, 4, 2, 1)  # B, H, W, K*K, G
        kernel = kernel.unsqueeze(-2)
                
        patches = F.unfold(x, kernel_size=self.kernel_size, dilation=1, padding=self.kernel_size // 2, stride=self.stride)
        patches = patches.view(batch_size, num_channels // self.group_number, 
                               self.group_number,
                               self.kernel_size * self.kernel_size, 
                               height, width)
        patches = patches.permute(0, 4, 5, 3, 1, 2)  # B, H, W, K*K, C//G, G
    
        output = kernel * patches
        output = output.sum(dim=3)  # B, H, W, C
        output = output.squeeze(4)        
        output =output.permute(0,3,1,2)
        return output, kernel
#MLBF
class MLBF_net(nn.Module):
    """
    MLBF_net (Multi-Lead Branching with Attention) is a neural network designed for multi-lead ECG classification.
    The network processes each lead through independent branches, and then applies an attention mechanism to weight
    the importance of each lead. This is useful for tasks like multi-lead ECG classification where different leads
    may contribute differently to the final prediction.

    Attributes:
    -----------
    nleads : int
        Number of ECG leads. Default is 12.
    num_classes : int
        Number of output classes for classification. Default is 9.
    branch{i} : Branchnet
        For each lead, an independent branch network (Branchnet) processes the input data for that lead. 
        These branches are dynamically created and assigned during initialization.
    attention : nn.Sequential
        Attention mechanism consisting of two linear layers and a Tanh activation, followed by a softmax function. 
        This mechanism calculates the importance scores for each lead.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the MLBF_net model. Processes each lead independently, applies attention, 
        and aggregates the results.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nleads, num_features), where `nleads` is the number of leads 
            (e.g., 12 for a standard ECG).

        Returns:
        --------
        x : torch.Tensor
            Output tensor after applying the attention-weighted aggregation of the branch outputs, with shape 
            (batch_size, num_classes), where `num_classes` is the number of target classes.
    """

    def __init__(self, nleads=12, num_classes=9):
        """
        Initializes the MLBF_net model.

        Parameters:
        -----------
        nleads : int, optional
            Number of ECG leads. Default is 12.
        num_classes : int, optional
            Number of output classes for classification. Default is 9.
        """
        super(MLBF_net, self).__init__()
        self.nleads = nleads
        
        # Create independent branches for each lead
        for i in range(nleads):
            setattr(self, "branch%d" % i, Branchnet(num_classes))
        
        # Define the attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features=nleads, out_features=nleads, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=nleads, out_features=nleads, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass of the MLBF_net model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nleads, num_features), where `nleads` corresponds to the number of leads.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, num_classes), which is the result of applying the attention-weighted
            aggregation of the branches' outputs across all leads.
        """
        branch_list = []
        
        # Pass each lead through its corresponding branch network
        for i in range(self.nleads):
            branch_list.append(getattr(self, "branch%d" % i)(x[:, i, :].unsqueeze(1)))
        
        # Stack the results from each lead
        x = torch.stack(branch_list, dim=2)
        
        # Apply attention mechanism
        score = self.attention(x)
        
        # Weight the branch outputs by the attention scores
        x = x * score
        
        # Aggregate across all leads
        x = torch.sum(x, dim=2)
        
        return x
    
class Branchnet(nn.Module):
    """
    Branchnet is a deep neural network used as an independent branch in a multi-branch model (like MLBF_net). 
    It consists of several 1D convolutional layers, dropout, a bi-directional GRU, and an attention mechanism, 
    followed by a fully connected layer for classification. The network is designed to process time-series data 
    such as single-lead ECG signals.

    Attributes:
    -----------
    layer0 : nn.Sequential
        The first sequential block of 1D convolutional layers with LeakyReLU activations and dropout.
    layer1 : nn.ModuleList
        A list of sequential blocks of 1D convolutional layers with LeakyReLU activations and dropout, repeated three times.
    layer2 : nn.Sequential
        A final sequential block of 1D convolutional layers with LeakyReLU activations and dropout.
    biGRU : nn.GRU
        A bidirectional GRU (Gated Recurrent Unit) layer that processes the output from the convolutional layers.
    attention : nn.Sequential
        Attention mechanism consisting of two linear layers, Tanh activation, and a softmax function to compute attention scores.
    fc : nn.Linear
        Fully connected layer for the final classification.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the Branchnet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, num_features), representing a single lead of the ECG signal.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, num_classes), representing the classification scores for each class.
    """

    def __init__(self, num_classes=9):
        """
        Initializes the Branchnet model.

        Parameters:
        -----------
        num_classes : int, optional
            Number of output classes for classification. Default is 9.
        """
        super(Branchnet, self).__init__()

        # Initial sequential block of 1D convolutions and dropout
        self.layer0 = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(12, 12, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(12, 12, kernel_size=24, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # A module list consisting of three sequential blocks of 1D convolutions and dropout
        self.layer1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(12, 12, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(12, 12, kernel_size=3, stride=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(12, 12, kernel_size=24, stride=2),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout(p=0.2)
                )
                for _ in range(3)
            ]
        )

        # Final sequential block of 1D convolutions and dropout
        self.layer2 = nn.Sequential(
            nn.Conv1d(12, 12, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(12, 12, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(12, 12, kernel_size=48, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Bidirectional GRU layer
        self.biGRU = nn.GRU(input_size=12, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features=24, out_features=24, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=24, out_features=24, bias=False),
            nn.Softmax(dim=1)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=24, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the Branchnet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, num_features), where `1` corresponds to the number of input channels 
            (for example, a single ECG lead).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, num_classes), which represents the classification scores for each class.
        """
        # Pass through the initial convolutional block
        x = self.layer0(x)
        
        # Pass through the module list of convolutional blocks
        for layer in self.layer1:
            x = layer(x)
        
        # Pass through the final convolutional block
        x = self.layer2(x)

        # Initialize hidden state for GRU
        h0 = torch.zeros(2, x.size(0), 12).to(x.device)  # 2 is for bidirectional

        # Pass through bidirectional GRU
        x_0, _ = self.biGRU(x.permute(0, 2, 1), h0)

        # Apply attention mechanism
        att_score = self.attention(x_0)
        x = x_0 * att_score

        # Sum across the time dimension
        x = torch.sum(x, dim=1)

        # Pass through the fully connected layer
        x = self.fc(x)

        return x

# ResUDense
class Bottleneck(nn.Module):
    """
    The Bottleneck class is a building block for DenseNet-like architectures. 
    It applies a series of batch normalization, 1D convolution, and ReLU activation to reduce the number 
    of channels, followed by a convolution that outputs a feature map with the desired growth rate. 
    This architecture is useful for increasing network depth without drastically increasing computation.

    Attributes:
    -----------
    bn1 : nn.BatchNorm1d
        Batch normalization layer applied to the input.
    conv1 : nn.Conv1d
        First 1D convolutional layer with kernel size 1 for reducing the number of channels.
    bn2 : nn.BatchNorm1d
        Batch normalization layer applied after the first convolution.
    conv2 : nn.Conv1d
        Second 1D convolutional layer with kernel size 3 for generating the output with growthRate channels.

    Methods:
    --------
    forward(x):
        Performs the forward pass, where the input is processed through the batch norm, ReLU, and convolution layers. 
        The output is concatenated with the input to form a dense connection.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, nChannels, length).

        Returns:
        --------
        out : torch.Tensor
            Output tensor with shape (batch_size, nChannels + growthRate, length), formed by concatenating the input 
            and the output of the bottleneck block.
    """

    def __init__(self, nChannels, growthRate):
        """
        Initializes the Bottleneck block.

        Parameters:
        -----------
        nChannels : int
            The number of input channels.
        growthRate : int
            The number of output channels (i.e., the growth rate).
        """
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)  # Concatenate input and output
        return out


class SingleLayer(nn.Module):
    """
    The SingleLayer class implements a single dense layer with batch normalization, 
    ReLU activation, and a 1D convolution. This layer is useful in dense architectures 
    like DenseNet, where each layer is connected to every subsequent layer.

    Attributes:
    -----------
    bn1 : nn.BatchNorm1d
        Batch normalization layer applied to the input.
    conv1 : nn.Conv1d
        A 1D convolutional layer with kernel size 3 for feature extraction.

    Methods:
    --------
    forward(x):
        Performs the forward pass, where the input is processed through the batch norm, ReLU, and convolution layers. 
        The output is concatenated with the input.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, nChannels, length).

        Returns:
        --------
        out : torch.Tensor
            Output tensor with shape (batch_size, nChannels + growthRate, length), 
            formed by concatenating the input and the output of the layer.
    """

    def __init__(self, nChannels, growthRate):
        """
        Initializes the SingleLayer block.

        Parameters:
        -----------
        nChannels : int
            The number of input channels.
        growthRate : int
            The number of output channels (i.e., the growth rate).
        """
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)  # Concatenate input and output
        return out
    
class Transition(nn.Module):
    """
    The Transition class reduces the number of channels and performs downsampling 
    (if required) in a DenseNet-like architecture. This is used between dense blocks to reduce complexity.

    Attributes:
    -----------
    bn1 : nn.BatchNorm1d
        Batch normalization layer applied to the input.
    conv1 : nn.Conv1d
        1D convolutional layer with kernel size 1 for reducing the number of channels.
    down : bool
        Flag indicating whether down-sampling (average pooling) should be applied.

    Methods:
    --------
    forward(x):
        Performs the forward pass, where the input is processed through batch norm, 
        ReLU, and convolution, followed by downsampling if required.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, nChannels, length).

        Returns:
        --------
        out : torch.Tensor
            Output tensor with shape (batch_size, nOutChannels, length // 2) if down-sampling is applied.
    """

    def __init__(self, nChannels, nOutChannels, down=False):
        """
        Initializes the Transition block.

        Parameters:
        -----------
        nChannels : int
            The number of input channels.
        nOutChannels : int
            The number of output channels.
        down : bool, optional
            Whether down-sampling (average pooling) should be applied. Default is False.
        """
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.down = down

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.down:
            out = F.avg_pool1d(out, 2)  # Down-sampling
        return out

class ResidualUBlock(nn.Module):
    """
    The ResidualUBlock class implements a U-Net-style block with residual connections. 
    It consists of encoder and decoder layers that downsample and upsample the input feature maps, 
    with skip connections to preserve fine-grained information.

    Attributes:
    -----------
    conv1 : nn.Conv1d
        Initial convolutional layer to process the input.
    bn1 : nn.BatchNorm1d
        Batch normalization layer after the first convolution.
    encoders : nn.ModuleList
        List of encoder layers that downsample the input feature maps.
    decoders : nn.ModuleList
        List of decoder layers that upsample the feature maps.
    bottleneck : nn.Sequential
        Bottleneck layer at the center of the U-Net.
    idfunc_0 : nn.AvgPool1d, optional
        Downsampling layer applied at the end of the U-Net if required.
    idfunc_1 : nn.Conv1d, optional
        1x1 convolution applied after downsampling if required.

    Methods:
    --------
    forward(x):
        Performs the forward pass through the encoder, bottleneck, and decoder layers. 
        Residual connections are added, and downsampling is applied if needed.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, out_ch, length).

        Returns:
        --------
        out : torch.Tensor
            Output tensor with the same shape as the input if no downsampling is applied. 
            If downsampling is applied, the output shape will have reduced length.
    """

    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        """
        Initializes the ResidualUBlock.

        Parameters:
        -----------
        out_ch : int
            The number of output channels.
        mid_ch : int
            The number of intermediate channels.
        layers : int
            The number of encoder-decoder layers in the U-Net.
        downsampling : bool, optional
            Whether downsampling should be applied at the end. Default is True.
        """
        super(ResidualUBlock, self).__init__()
        self.downsample = downsampling
        K = 9  # Kernel size
        P = (K - 1) // 2  # Padding

        self.conv1 = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            # Encoder
            if idx == 0:
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(out_ch, mid_ch, kernel_size=K, stride=2, padding=P, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(mid_ch, mid_ch, kernel_size=K, stride=2, padding=P, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

            # Decoder
            if idx == layers - 1:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(mid_ch * 2, out_ch, kernel_size=K, stride=2, padding=P, output_padding=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(mid_ch * 2, mid_ch, kernel_size=K, stride=2, padding=P, output_padding=1, bias=False),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []
        for layer in self.encoders:
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))

        out = out[..., :x_in.size(-1)]
        out += x_in

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out

def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    """
    Helper function to create a dense block consisting of multiple Bottleneck or SingleLayer units. 
    Each unit increases the number of channels by the growthRate.

    Parameters:
    -----------
    nChannels : int
        The number of initial input channels.
    growthRate : int
        The number of output channels for each Bottleneck or SingleLayer.
    nDenseBlocks : int
        The number of Bottleneck or SingleLayer units in the dense block.
    bottleneck : bool
        If True, the dense block will consist of Bottleneck units. If False, SingleLayer units will be used.

    Returns:
    --------
    nn.Sequential
        A sequential container consisting of the dense block layers.
    """
    layers = []
    for i in range(nDenseBlocks):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)

class ResU_Dense(nn.Module):
    """
    ResU_Dense is a deep neural network combining Residual U-blocks, DenseNet blocks, and multihead attention 
    for processing 1D sequential data such as ECG signals. The architecture includes initial convolutional layers, 
    residual U-blocks, dense blocks, transitions, and a multihead attention layer, followed by a fully connected 
    layer for classification.

    Attributes:
    -----------
    conv : nn.Conv1d
        Initial convolutional layer with kernel size 15 and stride 2 for reducing the dimensionality of the input.
    bn : nn.BatchNorm1d
        Batch normalization layer applied after the initial convolution.
    rub_0, rub_1, rub_2, rub_3 : ResidualUBlock
        Residual U-blocks that progressively downsample and upsample the input while preserving key features.
    dense1, dense2 : nn.Sequential
        Dense blocks that apply bottleneck layers and expand the number of channels.
    trans1, trans2 : Transition
        Transition layers that reduce the number of channels and apply downsampling.
    mha : nn.MultiheadAttention
        Multihead attention layer for capturing long-range dependencies in the sequence.
    pool : nn.AdaptiveMaxPool1d
        Max pooling layer to reduce the output size to a single value per channel.
    fc_1 : nn.Linear
        Fully connected layer for producing the final output.

    Methods:
    --------
    forward(x, quantize=False):
        Performs the forward pass of the ResU_Dense model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_ch, sequence_length), where `in_ch` is the number of input channels (e.g., 12 for ECG leads).
        quantize : bool, optional
            Whether to apply quantization padding to the input. Default is False.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, nOUT), representing the classification output.
    """

    def __init__(self, nOUT, in_ch=12, out_ch=256, mid_ch=64):
        """
        Initializes the ResU_Dense model.

        Parameters:
        -----------
        nOUT : int
            Number of output classes for classification.
        in_ch : int, optional
            Number of input channels. Default is 12 (e.g., for 12-lead ECG).
        out_ch : int, optional
            Number of output channels after the initial convolution. Default is 256.
        mid_ch : int, optional
            Number of intermediate channels in the residual U-blocks. Default is 64.
        """
        super(ResU_Dense, self).__init__()

        # Initial convolutional layer and batch normalization
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

        # Residual U-blocks
        self.rub_0 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=6)
        self.rub_1 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=5)
        self.rub_2 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_3 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        # Dense blocks and transitions
        growthRate = 12
        reduction = 0.5
        nChannels = out_ch
        nDenseBlocks = 16

        self.dense1 = _make_dense(nChannels, growthRate=growthRate, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = _make_dense(nChannels, growthRate=growthRate, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        self.trans2 = Transition(nChannels, out_ch)

        # Multihead attention layer and pooling
        self.mha = nn.MultiheadAttention(out_ch, 8)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layer for classification
        self.fc_1 = nn.Linear(out_ch, nOUT)

    def forward(self, x, quantize=False):
        """
        Forward pass of the ResU_Dense model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_ch, sequence_length).
        quantize : bool, optional
            Whether to apply quantization padding. Default is False.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, nOUT), representing the classification output.
        """
        # Pad input to ensure divisibility by 4
        if quantize:
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), 1000, dtype=torch.float16).to(x.device)), dim=2)
        else:
            x = torch.cat((x, torch.zeros(x.size(0), x.size(1), 1000).to(x.device)), dim=2)

        # Initial convolution and normalization
        x = F.leaky_relu(self.bn(self.conv(x)))

        # Pass through residual U-blocks
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.rub_2(x)
        x = self.rub_3(x)

        # Pass through dense blocks and transitions
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))

        # Apply dropout
        x = F.dropout(x, p=0.5, training=self.training)

        # Permute and pass through multihead attention
        x = x.permute(2, 0, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(1, 2, 0)

        # Pool and pass through fully connected layer
        x = self.pool(x).squeeze(2)
        x = self.fc_1(x)

        return x

# SGB
class DepthwiseSeparableConvolution(nn.Module):
    """
    DepthwiseSeparableConvolution is a 1D convolutional layer that performs depthwise separable convolution. 
    It divides the standard convolution into two parts: a depthwise convolution, where each input channel is 
    convolved independently, followed by a pointwise convolution that combines the outputs of the depthwise 
    convolution. This reduces the computational cost while retaining the ability to capture spatial patterns 
    and correlations between channels.

    Attributes:
    -----------
    depthwise : nn.Sequential
        A sequential block consisting of:
        - nn.Conv1d: Depthwise convolution with `nin` input channels, `nin` output channels, and `groups=nin`.
        - nn.BatchNorm1d: Batch normalization applied after the depthwise convolution.
        - nn.GELU: GELU activation function applied after batch normalization.
    pointwise : nn.Sequential
        A sequential block consisting of:
        - nn.Conv1d: Pointwise convolution with `nin` input channels and `nout` output channels.
        - nn.BatchNorm1d: Batch normalization applied after the pointwise convolution.
        - nn.GELU: GELU activation function applied after batch normalization.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the depthwise separable convolution.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nin, sequence_length), where `nin` is the number of input channels.

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, nout, sequence_length), where `nout` is the number of output channels.
    """

    def __init__(self, nin, nout, kernel_size=3, stride=1):
        """
        Initializes the DepthwiseSeparableConvolution layer.

        Parameters:
        -----------
        nin : int
            Number of input channels.
        nout : int
            Number of output channels.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride of the convolution. Default is 1.
        """
        super(DepthwiseSeparableConvolution, self).__init__()

        # Depthwise convolution: applies convolution separately on each input channel
        self.depthwise = nn.Sequential(
            nn.Conv1d(nin, nin, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=nin, bias=False),
            nn.BatchNorm1d(nin),
            nn.GELU(),
        )

        # Pointwise convolution: combines the output from depthwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(nout),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Forward pass of the DepthwiseSeparableConvolution layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nin, sequence_length).

        Returns:
        --------
        x : torch.Tensor
            Output tensor of shape (batch_size, nout, sequence_length).
        """
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Pointwise convolution
        x = self.pointwise(x)

        return x

class ECALayer(nn.Module):
    """
    ECALayer (Efficient Channel Attention Layer) is a module designed to capture channel-wise dependencies 
    without significant overhead. It uses global average pooling to extract global information and 
    a convolution operation with a small kernel size to capture cross-channel interaction. This module enhances 
    the representational power of the network by modulating channel responses adaptively.

    Attributes:
    -----------
    avg_pool : nn.AdaptiveAvgPool1d
        Global average pooling layer that reduces the input tensor to a single value per channel, summarizing the global information.
    conv : nn.Conv1d
        1D convolution applied to the pooled tensor for cross-channel interaction, with a kernel size defined by `k_size`.
    sigmoid : nn.Sigmoid
        Sigmoid activation function that scales the output between 0 and 1 to apply channel-wise attention.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the ECALayer, applying global average pooling, a convolution for channel attention, 
        and element-wise multiplication to modulate the input tensor.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channel, sequence_length), where `channel` is the number of channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, modulated by channel-wise attention.
    """

    def __init__(self, channel, k_size=3):
        """
        Initializes the ECALayer module.

        Parameters:
        -----------
        channel : int
            Number of input channels.
        k_size : int, optional
            Kernel size for the 1D convolution applied to the global feature descriptor. Default is 3.
        """
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling to generate the global feature descriptor
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 1D convolution
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for attention weights

    def forward(self, x):
        """
        Forward pass of the ECALayer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channel, sequence_length).

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, modulated by channel-wise attention.
        """
        # Global average pooling
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)  # Shape: (batch_size, 1, channel)

        # Convolution for channel attention
        y = self.conv(y).transpose(-1, -2)  # Shape: (batch_size, channel, 1)

        # Sigmoid activation for attention weights
        y = self.sigmoid(y)

        # Modulate the input tensor by channel-wise attention
        return x * y.expand_as(x)

class GhostModule(nn.Module):
    """
    GhostModule is a lightweight convolutional layer designed to reduce computational complexity and the number of parameters. 
    It generates feature maps in two steps: the first step uses standard convolutions to create a set of primary feature maps, 
    while the second step applies cheap depthwise convolutions to generate the rest of the feature maps, known as "ghost" features. 
    This module is useful for efficient neural networks, especially in resource-constrained environments.

    Attributes:
    -----------
    out_channels : int
        Number of output channels.
    primary_conv : nn.Sequential
        The first convolution block that creates the primary set of feature maps. It includes:
        - nn.Conv1d: A 1D convolution with `in_channels` input channels and `init_channels` output channels.
        - nn.BatchNorm1d: Batch normalization applied to the primary convolution output.
        - nn.GELU: Optional GELU activation function (or a no-op if `use_act` is False).
    cheap_operation : nn.Sequential
        The second block that generates the "ghost" feature maps using depthwise separable convolution. It includes:
        - nn.Conv1d: A depthwise convolution that expands the number of feature maps.
        - nn.BatchNorm1d: Batch normalization applied to the depthwise convolution output.
        - nn.GELU: Optional GELU activation function (or a no-op if `use_act` is False).

    Methods:
    --------
    forward(x):
        Performs the forward pass of the GhostModule, creating both primary and ghost feature maps, 
        then concatenating them to form the final output.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length), where `in_channels` is the number of input channels.

        Returns:
        --------
        out : torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length), where `out_channels` is the number of output channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, use_act=True):
        """
        Initializes the GhostModule.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Kernel size for the primary convolution. Default is 1.
        ratio : int, optional
            The ratio by which to reduce the number of channels in the primary convolution. Default is 2.
        dw_size : int, optional
            Kernel size for the depthwise convolution (used to generate the ghost features). Default is 3.
        stride : int, optional
            Stride for the primary convolution. Default is 1.
        use_act : bool, optional
            Whether to use the activation function (GELU) after the convolutions. Default is True.
        """
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)  # Primary feature map channels
        new_channels = init_channels * (ratio - 1)  # Ghost feature map channels

        # Primary convolution block
        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.GELU() if use_act else nn.Sequential(),  # Optional activation
        )

        # Cheap operation block (depthwise convolution)
        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, kernel_size=dw_size, stride=1,
                      padding=(dw_size - 1) // 2, groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.GELU() if use_act else nn.Sequential(),  # Optional activation
        )

    def forward(self, x):
        """
        Forward pass of the GhostModule.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
        --------
        out : torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Primary feature maps
        x1 = self.primary_conv(x)

        # Ghost feature maps
        x2 = self.cheap_operation(x1)

        # Concatenate primary and ghost feature maps
        out = torch.cat([x1, x2], dim=1)

        # Trim to the correct number of output channels
        return out[:, :self.out_channels, :]

class ShuffleBlock(nn.Module):
    """
    ShuffleBlock is a layer that performs channel shuffle, which is designed to help exchange information 
    between different groups of channels. This operation is typically used in architectures that involve group 
    convolutions to improve the flow of information across channel groups, such as in ShuffleNet. It reshapes 
    the input tensor, permutes the dimensions, and then reshapes it back to ensure that channels from different 
    groups are mixed.

    Attributes:
    -----------
    groups : int
        The number of groups into which the channels are divided for shuffling.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the ShuffleBlock, which shuffles the input channels to improve information flow 
        between groups.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length), where `channels` is the number of input channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, but with channels shuffled.
    """

    def __init__(self, groups):
        """
        Initializes the ShuffleBlock.

        Parameters:
        -----------
        groups : int
            The number of groups into which the channels will be divided for the shuffle operation.
        """
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        """
        Forward pass of the ShuffleBlock.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length), where `channels` is the number of input channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, but with channels shuffled between groups.
        """
        N, C, L = x.size()  # N: batch size, C: number of channels, L: sequence length
        g = self.groups

        # Reshape, permute, and then reshape back to shuffle the channels
        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)

class ShuffleGhostBottleneck(nn.Module):
    """
    ShuffleGhostBottleneck is a neural network module that combines the concepts of Ghost modules, depthwise separable convolutions, 
    channel shuffling, and optional Squeeze-and-Excite (SE) layers. This architecture is designed to be efficient in terms of computation 
    and parameters while retaining high performance for tasks such as 1D signal processing. It performs a bottleneck operation, where the input 
    is first projected into a smaller hidden space, processed through lightweight convolutions, and then projected back to the original 
    dimensionality. The residual connection allows the input to skip over the bottleneck, improving gradient flow during training.

    Attributes:
    -----------
    shuffle : nn.Module
        A ShuffleBlock that performs channel shuffling if `shuffle` is set to True. Otherwise, an identity operation.
    conv : nn.Sequential
        The main convolution block consisting of:
        - GhostModule: A 1D Ghost convolution that reduces the number of parameters by creating ghost feature maps.
        - Depthwise convolution: A depthwise separable convolution (optional, depending on stride).
        - ECALayer: A Squeeze-and-Excite layer for channel attention (optional, depending on `use_se`).
        - GhostModule: A final Ghost convolution with linear activation for projection to the output space.
    shortcut : Callable
        A shortcut connection for the residual path. If the input and output channels match and stride is 1, 
        the shortcut is an identity function. Otherwise, it uses a depthwise separable convolution to match the dimensions.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the ShuffleGhostBottleneck, applying the bottleneck operation and adding the residual connection.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length), where `in_channels` is the number of input channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length), where `out_channels` is the number of output channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, hidden_ratio=2, use_se=False, shuffle=False):
        """
        Initializes the ShuffleGhostBottleneck module.

        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Kernel size for the depthwise convolution.
        stride : int, optional
            Stride for the depthwise convolution. Default is 1.
        hidden_ratio : int, optional
            The ratio used to determine the number of hidden channels. Default is 2.
        use_se : bool, optional
            Whether to include the Squeeze-and-Excite layer for channel attention. Default is False.
        shuffle : bool, optional
            Whether to apply channel shuffling. Default is False.
        """
        super(ShuffleGhostBottleneck, self).__init__()
        assert stride in [1, 2], "Stride must be 1 or 2."

        # Calculate hidden channels
        hidden_channels = hidden_ratio * in_channels

        # Shuffle block for channel shuffling
        self.shuffle = ShuffleBlock(groups=2) if shuffle == 2 else nn.Sequential()

        # Main convolutional block
        self.conv = nn.Sequential(
            # Primary Ghost module (pointwise convolution)
            GhostModule(in_channels, hidden_channels, kernel_size=1, use_act=True),

            # Depthwise separable convolution (if stride is 2)
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=hidden_channels, bias=False),
                nn.BatchNorm1d(hidden_channels),
            ) if stride == 2 else nn.Sequential(),

            # Squeeze-and-Excite layer (optional)
            ECALayer(hidden_channels) if use_se else nn.Sequential(),

            # Final Ghost module (pointwise convolution, linear activation)
            GhostModule(hidden_channels, out_channels, kernel_size=1, use_act=False),
        )

        # Shortcut connection (identity if in_channels == out_channels and stride == 1)
        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = DepthwiseSeparableConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        """
        Forward pass of the ShuffleGhostBottleneck.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels, sequence_length).
        """
        # Apply the shortcut connection (residual path)
        residual = self.shortcut(x)

        # Apply the main convolution block (with optional shuffling)
        x = self.conv(self.shuffle(x))

        # Add the residual connection
        return x + residual

class SGB(nn.Module):
    """
    SGB (Shuffle Ghost Bottleneck) is a neural network architecture designed to efficiently process 1D data, 
    such as ECG signals, using ShuffleGhostBottleneck blocks. The network consists of several stages, 
    where each stage is composed of multiple ShuffleGhostBottleneck layers. The architecture begins with a projection layer, 
    followed by several stages of bottleneck layers, and ends with a global average pooling and classification layer.

    This architecture combines Ghost modules, shuffle operations, and optional Squeeze-and-Excite (SE) layers, 
    making it both lightweight and computationally efficient for tasks like signal classification.

    Attributes:
    -----------
    cfgs : list
        Configuration for each stage of the network. Each configuration contains parameters such as input/output channels, 
        kernel size, stride, hidden ratio, whether to use Squeeze-and-Excite (SE), and whether to shuffle channels.
    in_proj : nn.Sequential
        The initial projection layer that reduces the input dimension and prepares the input for the main network.
        It consists of:
        - nn.Conv1d: A 1D convolution to project the input channels to a defined number of channels.
        - nn.BatchNorm1d: Batch normalization layer.
        - nn.LeakyReLU: LeakyReLU activation function.
    layers : nn.Sequential
        A series of ShuffleGhostBottleneck layers applied to the input data.
    out_proj : nn.Sequential
        A projection layer that projects the final output to 1024 channels.
    gap : nn.Sequential
        Global average pooling and flattening layers, reducing the output to a single value per channel.
    classifier : nn.Linear
        A fully connected layer that outputs the final class predictions.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the SGB network, passing the input through the projection layers, 
        ShuffleGhostBottleneck layers, and finally through a global average pooling and classification layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, sequence_length), where `num_channels` is the number of input channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes), where `num_classes` is the number of classification labels.
    """

    def __init__(self, num_classes=9):
        """
        Initializes the SGB network.

        Parameters:
        -----------
        num_classes : int, optional
            Number of output classes for classification. Default is 9.
        """
        super(SGB, self).__init__()

        # Configuration for each stage of the network
        cfgs = [
            [
                [32, 64, 3, 2, 2, 1, 1],  # [in_channels, out_channels, kernel_size, stride, hidden_ratio, use_se, shuffle]
                [64, 64, 3, 1, 2, 1, 0],
                [64, 64, 3, 1, 2, 1, 0]
            ],
            [
                [64, 96, 3, 2, 2, 1, 1],
                [96, 96, 3, 1, 2, 1, 0],
                [96, 96, 3, 1, 2, 1, 0]
            ],
            [
                [96, 128, 3, 2, 2, 1, 1],
                [128, 128, 3, 1, 2, 1, 0],
            ]
        ]
        self.cfgs = cfgs

        # Input projection layer
        in_proj_channel = self.cfgs[0][0][0]
        self.in_proj = nn.Sequential(
            nn.Conv1d(12, in_proj_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_proj_channel),
            nn.LeakyReLU(inplace=True),
        )

        # Layers of ShuffleGhostBottleneck blocks
        layers = []
        for stage_cfg in self.cfgs:
            for in_c, out_c, k, s, r, use_se, shuffle in stage_cfg:
                layers.append(ShuffleGhostBottleneck(in_c, out_c, k, s, r, use_se, shuffle))
        self.layers = nn.Sequential(*layers)

        # Output projection layer
        out_proj_channel = self.cfgs[-1][-1][0]
        self.out_proj = nn.Sequential(
            nn.Conv1d(out_proj_channel, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        # Global Average Pooling and Flatten
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # Fully connected classifier
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        """
        Forward pass of the SGB network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, sequence_length), where `num_channels` is the number of input channels.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes).
        """
        # Initial projection
        x = self.in_proj(x)

        # Pass through ShuffleGhostBottleneck layers
        x = self.layers(x)

        # Final output projection
        x = self.out_proj(x)

        # Global average pooling and classification
        x = self.gap(x)
        x = self.classifier(x)

        return x

#cpsc_champion
def dot_product(x, kernel):
    result=torch.matmul(x, kernel.unsqueeze(0))
    return torch.squeeze(result, -1)

class cpsc_champion(nn.Module):
    """
    The `cpsc_champion` model is a deep neural network designed for 1D time-series classification tasks, such as ECG signal analysis. 
    The architecture includes multiple convolutional blocks, a bidirectional GRU (Gated Recurrent Unit), an attention mechanism, 
    and fully connected layers. The model processes multi-lead time-series data, applies feature extraction through convolution, 
    sequential modeling via the GRU, and attention to focus on important time steps before making the final classification.

    Attributes:
    -----------
    convblock : nn.ModuleList
        A list of convolutional blocks, each consisting of multiple 1D convolution layers, LeakyReLU activations, 
        and a dropout layer. There are four convolution blocks in total, each reducing the sequence length by half.
    convblock2 : nn.Sequential
        A final convolutional block that further processes the feature maps with additional 1D convolutions, 
        LeakyReLU activations, and dropout.
    bi_gru : nn.GRU
        A bidirectional GRU layer with 12 hidden units per direction, used to model sequential dependencies 
        in the input data.
    attention : nn.Sequential
        An attention mechanism applied after the GRU to focus on important time steps. It consists of two linear layers 
        with a Tanh activation in between, followed by a softmax activation to compute attention scores.
    dropout : nn.Dropout
        Dropout layer applied after the GRU and fully connected layers for regularization.
    batch_norm : nn.BatchNorm1d
        Batch normalization layer applied before the fully connected layer to normalize the output.
    fc : nn.Linear
        Fully connected layer that produces the final output for classification.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the model. The input is passed through convolutional blocks, 
        a GRU layer, an attention mechanism, and a fully connected layer to output classification scores.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, sequence_length), where `num_channels` is the number of input channels (e.g., 12 for ECG leads).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes), where `num_classes` is the number of output classes.
    """

    def __init__(self, seq_len, num_classes):
        """
        Initializes the `cpsc_champion` model.

        Parameters:
        -----------
        seq_len : int
            Length of the input sequence (e.g., number of time steps in the ECG signal).
        num_classes : int
            Number of output classes for classification.
        """
        super(cpsc_champion, self).__init__()
        
        # Four convolutional blocks
        self.convblock = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 24, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Dropout(0.2)
            ) for _ in range(4)
        ])

        # Final convolutional block
        self.convblock2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding='same'),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(12, 12, 3, padding='same'),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv1d(12, 12, 48, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Dropout(0.2)
        )

        # Bidirectional GRU
        self.bi_gru = nn.GRU(12, 12, batch_first=True, bidirectional=True)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(24, 24, bias=True),
            nn.Tanh(),
            nn.Linear(24, 24, bias=False),
            nn.Softmax(dim=1)
        )

        # Dropout and batch normalization
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(24)

        # Fully connected layer for classification
        self.fc = nn.Linear(24, num_classes)

    def forward(self, x):
        """
        Forward pass of the `cpsc_champion` model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_channels, sequence_length), where `num_channels` is the number of input channels (e.g., 12 for ECG leads).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes), representing the class scores.
        """
        a = x

        # Pass through the first four convolutional blocks
        for layer in self.convblock:
            x = layer(x)

        # Pass through the final convolutional block
        x = self.convblock2(x)

        # Reshape for GRU (batch_size, sequence_length, num_channels)
        x = x.permute(0, 2, 1)
        x, _ = self.bi_gru(x)

        # Apply activation and dropout
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.dropout(x)

        # Attention mechanism
        score = self.attention(x)
        x = score * x

        # Sum across the time dimension
        x = torch.sum(x, dim=1)

        # Apply batch normalization if there are multiple samples
        if a.size(0) != 1:
            x = self.batch_norm(x)

        # Apply activation and dropout
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.dropout(x)

        # Fully connected layer for classification
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model

def resnet10(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

def resnet12(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 1, 2, 1], **kwargs)
    return model

def resnet14(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 2, 2, 1], **kwargs)
    return model

def resnet22(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 3, 3, 2], **kwargs)
    return model

def resnet26(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 3, 3, 3], **kwargs)
    return model

def resnet28(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 3, 4, 3], **kwargs)
    return model

def resnet30(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 4, 3], **kwargs)
    return model

def resnet40(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 6, 7, 3], **kwargs)
    return model

def resnet38(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 6, 6, 3], **kwargs)
    return model

def resnet42(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 7, 7, 3], **kwargs)
    return model

# Mamba
class Mamba(nn.Module):
    """
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces O(L).
    This model implements a sequence processing architecture based on the Mamba method, designed to handle 
    long sequences efficiently in linear time complexity O(L). The architecture leverages residual blocks, 
    token embeddings, positional embeddings, and selective state space models for tasks like time-series 
    classification or forecasting.

    Reference:
    ----------
    Paper: https://arxiv.org/abs/2312.00752
    Implementation: https://github.com/johnma2006/mamba-minimal/

    Attributes:
    -----------
    d_inner : int
        The expanded dimension of the model, calculated as `d_model * expand`.
    dt_rank : int
        The rank for dynamic temporal state space, calculated as `ceil(d_model / 16)`.
    token_embedding : TokenEmbedding
        The embedding layer that converts input tokens (e.g., time-series data) into a dense representation of size `d_model`.
    position_embedding : PositionalEmbedding
        The positional embedding layer that encodes positional information into the token embeddings.
    input_layer : nn.Conv1d
        Initial convolution layer to adjust the number of input channels to the model's `d_model`.
    layers : nn.ModuleList
        A list of residual blocks, each consisting of convolutions and feed-forward layers for feature extraction.
    norm : RMSNorm
        Root mean square normalization applied before the output layer.
    out_layer : nn.Linear
        A linear layer that maps the final model representation to the output classes or dimensions.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the Mamba model, passing the input through token and positional embeddings, 
        a series of residual blocks, normalization, and finally a fully connected output layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_channels), where `sequence_length` is the length of the time-series data.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, c_out), representing the model's predictions or classifications.
    """

    def __init__(self, d_model, expand, enc_in, c_out, d_conv, d_ff, e_layers=2, dropout=0.1):
        """
        Initializes the Mamba model.

        Parameters:
        -----------
        d_model : int
            The base dimension of the model, which controls the size of the internal representations.
        expand : int
            Expansion factor to increase the internal dimension (`d_inner = d_model * expand`).
        enc_in : int
            Number of input channels or features (e.g., 12 for 12-lead ECG).
        c_out : int
            Number of output classes or regression targets.
        d_conv : int
            Dimension for the convolutional layers inside the residual blocks.
        d_ff : int
            Dimension of the feed-forward layers inside the residual blocks.
        e_layers : int, optional
            Number of residual layers. Default is 2.
        dropout : float, optional
            Dropout rate for regularization. Default is 0.1.
        """
        super(Mamba, self).__init__()

        # Calculate internal dimensions
        sequence_length = 15000
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # Token and positional embedding layers
        self.token_embedding = TokenEmbedding(sequence_length, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=sequence_length)

        # Initial convolution to adapt input channels to d_model
        self.input_layer = nn.Conv1d(enc_in, d_model, kernel_size=15, stride=2, padding=7, bias=False)

        # Residual blocks for sequence modeling
        self.layers = nn.ModuleList([ResidualBlock(d_model, d_conv, d_ff, self.d_inner, self.dt_rank) for _ in range(e_layers)])

        # Normalization and output layers
        self.norm = RMSNorm(d_model)
        self.out_layer = nn.Linear(d_model, c_out, bias=False)

    def forward(self, x):
        """
        Forward pass of the Mamba model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, num_channels).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, c_out), representing the model's predictions or classifications.
        """
        # Token embedding
        x = self.token_embedding(x)

        # Pass through residual layers
        for layer in self.layers:
            x = layer(x)

        # Normalize and apply output layer
        x = self.norm(x)
        x_out = self.out_layer(x.mean(dim=1))  # Mean pooling over sequence length

        return x_out

    
class ResidualBlock(nn.Module):
    """
    The `ResidualBlock` class implements a basic residual block with a MambaBlock for sequence modeling. 
    Residual connections are used to enhance gradient flow during training, allowing the model to learn deeper representations 
    without the vanishing gradient problem. The block consists of a normalization layer followed by a MambaBlock, 
    which applies selective state-space modeling.

    Attributes:
    -----------
    mixer : MambaBlock
        A MambaBlock that applies the main sequence modeling operations using convolutions and feed-forward layers.
    norm : RMSNorm
        Root mean square normalization applied to the input before passing it to the `mixer`.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the residual block, applying normalization followed by the MambaBlock, 
        and adding the input (residual connection) to the output of the block.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model), where `d_model` is the number of input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, with the residual connection applied.
    """

    def __init__(self, d_model, d_conv, d_ff, d_inner, dt_rank):
        """
        Initializes the `ResidualBlock` module.

        Parameters:
        -----------
        d_model : int
            The dimension of the model's internal representations.
        d_conv : int
            Dimension for the convolutional layers inside the MambaBlock.
        d_ff : int
            Dimension of the feed-forward layers inside the MambaBlock.
        d_inner : int
            Expanded dimension of the model, typically larger than `d_model`.
        dt_rank : int
            Rank for the dynamic temporal state space.
        """
        super(ResidualBlock, self).__init__()

        # MambaBlock for sequence modeling
        self.mixer = MambaBlock(d_model, d_conv, d_ff, d_inner, dt_rank)

        # Normalization layer (RMS normalization)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """
        Forward pass of the `ResidualBlock`.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, after applying the residual connection.
        """
        # Normalize input, apply MambaBlock, and add the residual connection
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    """
    MambaBlock is a core component of the Mamba architecture, designed for efficient sequence modeling. 
    It implements a selective state space mechanism, combining convolutional processing, linear projections, 
    and state-space models. The MambaBlock efficiently handles long sequences using selective updates and 
    state-specific delta calculations, which are inspired by recent advances in linear-time sequence models.

    Attributes:
    -----------
    d_inner : int
        The expanded dimension used for internal processing within the block.
    dt_rank : int
        The rank of the dynamic temporal state space used in the state-space model.
    in_proj : nn.Linear
        A linear layer that projects the input sequence to a higher-dimensional space (`2 * d_inner`).
    conv1d : nn.Conv1d
        A depthwise 1D convolution layer that processes the input sequence with `d_inner` channels.
    x_proj : nn.Linear
        A linear layer that projects the input sequence into the dynamic temporal state space, 
        generating input-specific deltas (`delta`) and parameters (`B` and `C`).
    dt_proj : nn.Linear
        A linear layer that projects `delta` to match the dimensionality of the state space (`d_inner`).
    A_log : nn.Parameter
        A logarithmic parameter representing the state matrix `A`, which is used in the selective scan process.
    D : nn.Parameter
        A parameter representing the diagonal scaling applied to the final output.
    out_proj : nn.Linear
        A linear layer that projects the processed sequence back to the model dimension (`d_model`).

    Methods:
    --------
    forward(x):
        Performs the forward pass of the MambaBlock, which includes input projection, convolutional processing, 
        and the selective state-space mechanism. The output is computed by applying the state-space model 
        to the convolutional features and combining the result with a residual connection.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model), where `d_model` is the number of input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_model), after applying the selective state-space mechanism and residual connection.
    
    ssm(x):
        Implements the state-space model (SSM) as described in Algorithm 2 of the Mamba paper. The method computes the 
        sequence-specific state transitions based on input features and the state matrices `A`, `B`, and `C`.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_inner).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_inner), after applying the state-space updates.
    
    selective_scan(u, delta, A, B, C, D):
        The selective scan algorithm performs the sequential state-space update across the input sequence. 
        It applies the state transition matrices and input-specific deltas to compute the final output.

        Parameters:
        -----------
        u : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_inner), representing the sequence input to the state-space model.
        delta : torch.Tensor
            Input-specific deltas for the state transition matrix `A`.
        A : torch.Tensor
            The state transition matrix, controlling how the state evolves over time.
        B : torch.Tensor
            The input matrix, controlling how the input influences the state evolution.
        C : torch.Tensor
            The output matrix, controlling how the state is mapped to the output.
        D : torch.Tensor
            Diagonal scaling factor applied to the final output.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_inner), representing the final result of the state-space updates.
    """

    def __init__(self, d_model, d_conv, d_ff, d_inner, dt_rank):
        """
        Initializes the MambaBlock.

        Parameters:
        -----------
        d_model : int
            The dimensionality of the input sequence (number of input features).
        d_conv : int
            Kernel size for the depthwise 1D convolution.
        d_ff : int
            Dimension of the feed-forward layer inside the block.
        d_inner : int
            Expanded dimension for internal processing, typically larger than `d_model`.
        dt_rank : int
            The rank for the dynamic temporal state space.
        """
        super(MambaBlock, self).__init__()

        # Internal and dynamic temporal state-space dimensions
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        # Linear projection to expand input into 2 * d_inner dimensions
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise convolution for local feature extraction
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # Linear projection for dynamic temporal state space
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_ff * 2, bias=False)

        # Delta projection for state-space updates
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Logarithmic parameter for state matrix A and scaling parameter D
        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Forward pass of the MambaBlock.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_model).
        """
        (b, l, d) = x.shape

        # Project input to 2 * d_inner dimensions
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        # Apply convolution along the sequence dimension
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]  # Ensure output matches input length
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        # Apply state-space model
        y = self.ssm(x)

        # Combine with residual connection
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """
        Implements the state-space model (SSM) from Algorithm 2 of the Mamba paper.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_inner).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_inner).
        """
        (d_in, n) = self.A_log.shape

        # Compute state-space matrices A and D
        A = -torch.exp(self.A_log.float())  # State transition matrix
        D = self.D.float()  # Scaling parameter

        # Compute delta, B, and C for state-space update
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)

        # Compute delta projection
        delta = F.softplus(self.dt_proj(delta))

        # Apply selective scan for state-space update
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """
        Performs selective scan as part of the state-space model update. Sequentially updates the state 
        across the input sequence using discretized state-space matrices.

        Parameters:
        -----------
        u : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_inner).
        delta : torch.Tensor
            Input-specific delta for the state transition matrix `A`.
        A : torch.Tensor
            State transition matrix controlling state evolution.
        B : torch.Tensor
            Input matrix influencing state transitions.
        C : torch.Tensor
            Output matrix mapping states to output space.
        D : torch.Tensor
            Diagonal scaling parameter applied to the final output.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_inner), after state-space updates.
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Compute discretized A and B matrices
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        # Initialize state and perform selective scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization) is a normalization technique that normalizes the input based on 
    the root mean square of the elements along the last dimension. Unlike traditional layer normalization, RMSNorm 
    avoids centering the input (i.e., no mean subtraction), making it more efficient in certain cases.

    RMSNorm scales the normalized output by a learnable weight parameter, and a small epsilon is added to prevent 
    division by zero during the normalization process.

    Attributes:
    -----------
    eps : float
        A small constant added to the denominator to prevent division by zero. Default is 1e-5.
    weight : nn.Parameter
        A learnable scaling factor applied to the normalized output, with a shape of `(d_model,)`, where `d_model` 
        is the number of input features.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the RMSNorm layer, normalizing the input based on the root mean square of the 
        elements along the last dimension and scaling it by the learnable weight parameter.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model), where `d_model` is the number of input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as the input, with RMS normalization applied.
    """

    def __init__(self, d_model, eps=1e-5):
        """
        Initializes the RMSNorm layer.

        Parameters:
        -----------
        d_model : int
            The number of input features (dimensionality) for each element in the sequence.
        eps : float, optional
            A small constant added to the denominator to prevent division by zero. Default is 1e-5.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # Learnable scaling factor for normalization

    def forward(self, x):
        """
        Forward pass of the RMSNorm layer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, d_model), where `d_model` is the number of input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, d_model), with RMS normalization applied.
        """
        # Compute the root mean square (RMS) of the input tensor along the last dimension
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Normalize and scale the input by the learnable weight
        output = x * rms * self.weight

        return output


#retnet
class SimpleRetention(nn.Module):
    """
    SimpleRetention implements a retention mechanism for sequence modeling, based on the paper 
    "Retentive Network: A Successor to Transformer for Large Language Models" 
    (https://arxiv.org/pdf/2307.08621.pdf). This mechanism leverages key-value pairs with retention-based 
    positional scaling to model long sequences efficiently.

    The class provides three forward methods: 
    - `forward`: Standard full-sequence retention mechanism.
    - `forward_recurrent`: Recurrent retention for processing sequences incrementally.
    - `forward_chunkwise`: Chunkwise processing for long sequences, handling sequence fragments separately.

    Attributes:
    -----------
    hidden_size : int
        Dimensionality of the input embeddings and hidden layers.
    head_size : int
        Dimensionality of the query and key projections. Defaults to `hidden_size` if not provided.
    v_dim : int
        Dimensionality of the value projection. Doubled if `double_v_dim` is set to True.
    gamma : float
        The decay factor applied to retention scores over time steps.
    W_Q : nn.Parameter
        Learnable weights for the query projection.
    W_K : nn.Parameter
        Learnable weights for the key projection.
    W_V : nn.Parameter
        Learnable weights for the value projection.
    xpos : XPOS
        Positional embedding layer used for modifying the query and key based on their position in the sequence.
    quantize : bool
        If True, quantization is applied to the retention scores for efficient computation.

    Methods:
    --------
    forward(X):
        Applies the standard retention mechanism to the entire input sequence.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_size), where `hidden_size` is the feature size.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_size), after applying retention-based attention.
    
    forward_recurrent(x_n, s_n_1, n):
        Processes input in a recurrent manner, updating the state incrementally for each time step.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor at the current time step, of shape (batch_size, hidden_size).
        s_n_1 : torch.Tensor
            Retention state from the previous time step, of shape (batch_size, head_size, v_dim).
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, torch.Tensor
            - The updated retention score for the current time step, of shape (batch_size, hidden_size).
            - The updated state, to be passed to the next time step.

    forward_chunkwise(x_i, r_i_1, i):
        Processes input in chunks for long sequences, handling one chunk at a time.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_size).
        r_i_1 : torch.Tensor
            Retention state from the previous chunk, of shape (batch_size, head_size, v_dim).
        i : int
            The current chunk index.

        Returns:
        --------
        torch.Tensor, torch.Tensor
            - The retention score for the current chunk, of shape (batch_size, chunk_size, hidden_size).
            - The updated retention state to be passed to the next chunk.

    _get_D(sequence_length):
        Computes the decay matrix `D` for a sequence of the given length based on the gamma decay factor. 
        The matrix controls how retention scores decay over time steps.

        Parameters:
        -----------
        sequence_length : int
            The length of the sequence for which to compute the decay matrix.

        Returns:
        --------
        torch.Tensor
            Decay matrix `D` of shape (sequence_length, sequence_length).
    """

    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False, quantize=False):
        """
        Initializes the SimpleRetention class.

        Parameters:
        -----------
        hidden_size : int
            Dimensionality of the input embeddings and hidden layers.
        gamma : float
            Decay factor that controls how retention scores decay over time steps.
        head_size : int, optional
            Dimensionality of the query and key projections. Defaults to `hidden_size` if not provided.
        double_v_dim : bool, optional
            If True, the value dimension (`v_dim`) is doubled for larger capacity. Default is False.
        quantize : bool, optional
            If True, quantization is applied to the retention scores for efficient computation. Default is False.
        """
        super(SimpleRetention, self).__init__()

        self.quantize = True if quantize else False
        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size

        self.head_size = head_size
        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        # Learnable query, key, and value projections
        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)

        # Positional embeddings
        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Forward pass of the retention mechanism, applied to the entire sequence.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device)

        # Compute queries and keys
        Q = X @ self.W_Q
        K = X @ self.W_K

        # Apply positional embeddings
        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        # Compute values
        V = X @ self.W_V

        # Apply retention mechanism with decay matrix
        if self.quantize:
            ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0).half()
        else:
            ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return ret @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent version of the forward pass, processing one time step at a time.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor at the current time step, of shape (batch_size, hidden_size).
        s_n_1 : torch.Tensor
            Retention state from the previous time step, of shape (batch_size, head_size, v_dim).
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, torch.Tensor
            Updated retention score and state for the current time step.
        """
        Q = x_n @ self.W_Q
        K = x_n @ self.W_K

        # Apply positional embeddings for the current time step
        Q = self.xpos(Q, n + 1)
        K = self.xpos(K, n + 1, downscale=True)

        # Compute value and update retention state
        V = x_n @ self.W_V
        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)

        return (Q @ s_n), s_n

    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise processing for long sequences, handling one chunk at a time.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_size).
        r_i_1 : torch.Tensor
            Retention state from the previous chunk, of shape (batch_size, head_size, v_dim).
        i : int
            Current chunk index.

        Returns:
        --------
        torch.Tensor, torch.Tensor
            Updated retention score for the current chunk and updated retention state.
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = x_i @ self.W_Q
        K = x_i @ self.W_K

        # Apply positional embeddings for the chunk
        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V

        # Compute retention across chunks
        r_i = (K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1
        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V

        # Compute cross-chunk retention
        e = torch.zeros(batch, chunk_size, 1)
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)

        cross_chunk = (Q @ r_i_1) * e
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        """
        Computes the decay matrix `D` for a given sequence length. The matrix controls how retention scores 
        decay over time steps based on the gamma decay factor.

        Parameters:
        -----------
        sequence_length : int
            The length of the sequence.

        Returns:
        --------
        torch.Tensor
            A decay matrix `D` of shape (sequence_length, sequence_length).
        """
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Compute decay matrix with masking
        D = (self.gamma ** (n - m)) * (n >= m).float()
        D[D != D] = 0  # Replace NaNs with 0
        return D

class MultiScaleRetention(nn.Module):
    """
    MultiScaleRetention implements a multi-scale retention mechanism, inspired by the paper 
    "Retentive Network: A Successor to Transformer for Large Language Models" 
    (https://arxiv.org/pdf/2307.08621.pdf). It utilizes multiple retention heads with different decay factors 
    (`gammas`) to capture information at different scales, enhancing the model's ability to handle sequences 
    with varying levels of temporal dependencies.

    The class provides three forward methods:
    - `forward`: Standard multi-scale retention applied to the entire sequence.
    - `forward_recurrent`: Recurrent processing of sequences for incremental updates.
    - `forward_chunkwise`: Chunkwise processing for long sequences.

    Attributes:
    -----------
    hidden_size : int
        The dimensionality of the input embeddings and hidden layers.
    v_dim : int
        Dimensionality of the value projection. Doubled if `double_v_dim` is set to True.
    heads : int
        Number of attention heads for multi-scale retention.
    head_size : int
        Dimensionality of the query and key projections for each head, calculated as `hidden_size // heads`.
    head_v_dim : int
        Dimensionality of the value projection for each head, optionally doubled.
    gammas : list
        List of decay factors (`gamma`) for each head, computed logarithmically to capture multi-scale dependencies.
    W_G : nn.Parameter
        Learnable projection matrix for the input (`hidden_size x v_dim`), used to scale the attention output.
    W_O : nn.Parameter
        Learnable projection matrix for the output (`v_dim x hidden_size`), used to map the result back to the input dimension.
    group_norm : nn.GroupNorm
        Group normalization applied across the concatenated output of all retention heads.
    retentions : nn.ModuleList
        A list of `SimpleRetention` modules, each representing an individual retention mechanism for one head.

    Methods:
    --------
    forward(X):
        Applies the multi-scale retention mechanism to the entire input sequence.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_size), where `hidden_size` is the feature size.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_size), after applying multi-scale retention.
    
    forward_recurrent(x_n, s_n_1s, n):
        Recurrent processing of the input sequence, updating the retention state incrementally for each time step.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor for the current time step, of shape (batch_size, hidden_size).
        s_n_1s : list of torch.Tensor
            Retention states from the previous time step, one per head, each of shape (batch_size, head_size, head_v_dim).
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            - Updated retention score for the current time step, of shape (batch_size, hidden_size).
            - Updated states for the next time step, one per head.
    
    forward_chunkwise(x_i, r_i_1s, i):
        Chunkwise processing of the input sequence, handling one chunk at a time.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_size).
        r_i_1s : list of torch.Tensor
            Retention states from the previous chunk, one per head, each of shape (batch_size, head_size, head_v_dim).
        i : int
            Current chunk index.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            - Updated retention score for the current chunk, of shape (batch_size, chunk_size, hidden_size).
            - Updated states for the next chunk, one per head.
    """

    def __init__(self, hidden_size, heads, double_v_dim=False, quantize=False):
        """
        Initializes the MultiScaleRetention module.

        Parameters:
        -----------
        hidden_size : int
            The dimensionality of the input embeddings and hidden layers.
        heads : int
            The number of attention heads for multi-scale retention.
        double_v_dim : bool, optional
            If True, the value dimension (`v_dim`) is doubled for larger capacity. Default is False.
        quantize : bool, optional
            If True, quantization is applied to the retention scores for efficient computation. Default is False.
        """
        super(MultiScaleRetention, self).__init__()

        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size

        # Compute the gamma decay factors for each head (multi-scale retention)
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        # Swish activation function
        self.swish = lambda x: x * torch.sigmoid(x)

        # Learnable projection matrices for input and output
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)

        # Group normalization applied to the concatenated output of all heads
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        # List of SimpleRetention mechanisms, one per head
        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim, quantize) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        Forward pass of the multi-scale retention mechanism applied to the entire sequence.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, sequence_length, hidden_size), after applying multi-scale retention.
        """
        # Apply each retention head to the input sequence
        Y = [self.retentions[i](X) for i in range(self.heads)]

        # Concatenate the output of all retention heads
        Y = torch.cat(Y, dim=2)

        # Apply group normalization
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        # Final output projection
        return (self.swish(X @ self.W_G) * Y) @ self.W_O

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        Recurrent version of the forward pass, processing one time step at a time.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor for the current time step, of shape (batch_size, hidden_size).
        s_n_1s : list of torch.Tensor
            Retention states from the previous time step, one per head.
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            Updated retention score for the current time step, and updated states for the next time step.
        """
        # Apply each retention head to the current time step
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(x_n[:, :, :], s_n_1s[i], n)
            Y.append(y)
            s_ns.append(s_n)

        # Concatenate the output of all heads
        Y = torch.cat(Y, dim=2)

        # Apply group normalization
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        # Final output projection
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        Chunkwise processing of the input sequence, handling one chunk at a time.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_size).
        r_i_1s : list of torch.Tensor
            Retention states from the previous chunk, one per head.
        i : int
            Current chunk index.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            Updated retention score for the current chunk, and updated states for the next chunk.
        """
        # Apply each retention head to the current chunk
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(x_i[:, :, :], r_i_1s[j], i)
            Y.append(y)
            r_is.append(r_i)

        # Concatenate the output of all heads
        Y = torch.cat(Y, dim=2)

        # Apply group normalization
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        # Final output projection
        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is

class RetNet(nn.Module):
    """
    RetNet is a deep network that leverages the multi-scale retention mechanism for sequence modeling, designed to handle long 
    sequences efficiently. RetNet stacks multiple layers of `MultiScaleRetention` blocks, with feed-forward networks (FFN) 
    and layer normalization to build a robust model for tasks such as classification or regression. 

    This architecture applies retention mechanisms at multiple scales across layers, making it suitable for tasks that 
    require capturing both short-term and long-term dependencies.

    Attributes:
    -----------
    layers : int
        Number of layers (blocks) in the RetNet model.
    hidden_dim : int
        The dimensionality of the hidden layers and embeddings in the model.
    ffn_size : int
        The dimensionality of the hidden layer in the feed-forward network.
    heads : int
        Number of attention heads used in each retention block.
    v_dim : int
        Dimensionality of the value projection, which can be optionally doubled.
    retentions : nn.ModuleList
        A list of `MultiScaleRetention` modules, each representing a retention block in the network.
    ffns : nn.ModuleList
        A list of feed-forward networks applied after each retention block, consisting of two linear layers with a GELU activation.
    layer_norms_1 : nn.ModuleList
        A list of `LayerNorm` modules, applied before each retention block.
    layer_norms_2 : nn.ModuleList
        A list of `LayerNorm` modules, applied before each feed-forward network (FFN).
    token_embedding : TokenEmbedding
        A token embedding layer that maps the input sequence to the hidden dimension.
    fc : nn.Linear
        A fully connected layer that maps the hidden representation to the final output (e.g., class logits).
    sequence_length : int
        The length of the input sequence.
    features : int
        The number of features in the input sequence (e.g., the number of input channels or embeddings).

    Methods:
    --------
    forward(X):
        Performs the forward pass of the RetNet model, applying retention blocks, feed-forward layers, and layer normalization.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, features), where `features` is the number of input features.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes), representing the class logits or regression targets.
    
    forward_recurrent(x_n, s_n_1s, n):
        Processes the input sequence in a recurrent manner, handling one time step at a time.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor for the current time step, of shape (batch_size, hidden_dim).
        s_n_1s : list of torch.Tensor
            Retention states from the previous time step, one per layer.
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            - The updated hidden state for the current time step, of shape (batch_size, hidden_dim).
            - The updated retention states for each layer.
    
    forward_chunkwise(x_i, r_i_1s, i):
        Processes the input sequence in chunks, useful for handling long sequences.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_dim).
        r_i_1s : list of torch.Tensor
            Retention states from the previous chunk, one per layer.
        i : int
            Current chunk index.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            - The updated hidden state for the current chunk, of shape (batch_size, chunk_size, hidden_dim).
            - The updated retention states for each layer.
    """

    def __init__(self, layers, hidden_dim, ffn_size, heads, sequence_length, features, num_classes, double_v_dim=False):
        """
        Initializes the RetNet model.

        Parameters:
        -----------
        layers : int
            Number of layers (blocks) in the model.
        hidden_dim : int
            Dimensionality of the hidden layers and embeddings.
        ffn_size : int
            Dimensionality of the hidden layer in the feed-forward network.
        heads : int
            Number of attention heads used in the retention blocks.
        sequence_length : int
            Length of the input sequence.
        features : int
            Number of input features or channels (e.g., the number of input dimensions for each token).
        num_classes : int
            Number of output classes for classification or regression targets.
        double_v_dim : bool, optional
            If True, the value dimension in the retention blocks is doubled for larger capacity. Default is False.
        """
        super(RetNet, self).__init__()

        # Initialize model attributes
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        # Initialize MultiScaleRetention blocks for each layer
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])

        # Initialize feed-forward networks (FFN) for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])

        # Initialize layer normalization modules
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

        # Token embedding layer to project input into hidden dimension
        self.token_embedding = TokenEmbedding(sequence_length, hidden_dim)

        # Fully connected layer for classification output
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Store sequence length and features
        self.sequence_length = sequence_length
        self.features = features

    def forward(self, X):
        """
        Forward pass of the RetNet model.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, features).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes).
        """
        batch_size, sequence_length, features = X.shape

        # Apply token embedding
        X = self.token_embedding(X)

        # Apply retention blocks and feed-forward layers across all layers
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        # Compute final output using the fully connected layer
        X = self.fc(X.mean(dim=1))
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        Recurrent version of the forward pass, processing one time step at a time.

        Parameters:
        -----------
        x_n : torch.Tensor
            Input tensor for the current time step, of shape (batch_size, hidden_dim).
        s_n_1s : list of torch.Tensor
            Retention states from the previous time step, one per layer.
        n : int
            Current time step.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            Updated hidden state and retention states for the current time step.
        """
        s_ns = []

        # Process input through retention blocks and FFN for each layer
        for i in range(self.layers):
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        Chunkwise processing of the input sequence.

        Parameters:
        -----------
        x_i : torch.Tensor
            Input tensor for the current chunk, of shape (batch_size, chunk_size, hidden_dim).
        r_i_1s : list of torch.Tensor
            Retention states from the previous chunk, one per layer.
        i : int
            Current chunk index.

        Returns:
        --------
        torch.Tensor, list of torch.Tensor
            Updated hidden state and retention states for the current chunk.
        """
        r_is = []

        # Process input through retention blocks and FFN for each chunk
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is
