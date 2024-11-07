import torch
import torch.optim as optim
import torch.nn as nn

from torchaudio.models.tacotron2 import _get_mask_from_lengths, Tacotron2
from torchaudio.pipelines import TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH


class MyTacotron2(Tacotron2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        tokens,
        token_lengths,
        mel_specgram,
        mel_specgram_lengths,
    ):
        embedded_inputs = self.embedding(tokens).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, token_lengths)
        mel_specgram, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_specgram, memory_lengths=token_lengths
        )

        mel_specgram_postnet = self.postnet(mel_specgram)
        mel_specgram_postnet = mel_specgram + mel_specgram_postnet

        if self.mask_padding:
            mask = _get_mask_from_lengths(mel_specgram_lengths)
            mask = mask.expand(self.n_mels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            mel_specgram.masked_fill_(mask, 0.0)
            mel_specgram_postnet.masked_fill_(mask, 0.0)
            gate_outputs.masked_fill_(mask[:, 0, :], 1e3)

        return embedded_inputs, encoder_outputs, mel_specgram, mel_specgram_postnet, gate_outputs, alignments


device = "cuda" if torch.cuda.is_available() else "cpu"

student = MyTacotron2(
    mask_padding=False,
    n_mels=80,
    n_symbol=148,
    n_frames_per_step=1,
    symbol_embedding_dim=512,  # Reduce to 128 or MLE
    encoder_embedding_dim=512,  # Reduce for dist
    encoder_n_convolution=3,
    decoder_rnn_dim=1024,
    attention_rnn_dim=1024,  # Reduce for dist
    attention_hidden_dim=128,  # Reduce for dist
    attention_location_n_filter=32,  # Reduce for dist
    prenet_dim=256,  # Reduce for dist
    postnet_n_convolution=5,  # Reduce for dist
    postnet_embedding_dim=512,  # Reduce for dist
)

teacher = TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_tacotron2().to(device)
processor = TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()


def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")


def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_knowledge_distillation(teacher=teacher, student=student, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_deep = test(teacher, test_loader, device)
test_accuracy_light_ce_and_kd = test(student, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")
