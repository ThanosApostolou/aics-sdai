import torch
from classes.MultiHeadAttention import MultiHeadAttention


def main():
    print("start main")
    d_model: int = 64
    num_heads: int = 8

    # consider the following two_dimensional tensor:
    t = torch.tensor([[0, 1, 2, 3],
                      [4, 5, 6, 7],
                      [8, 9, 10, 11]])
    

    print('t.size()', t.size())
    print('t.is_contiguous()', t.is_contiguous())
    print('t.stride()', t.stride())

    t = t.transpose(0, 1)
    print('t', t)
    print('t.size()', t.size())
    print('t.is_contiguous()', t.is_contiguous())
    print('t.stride()', t.stride())

    t = t.contiguous()
    print('t.stride()', t.stride())
    # model = MultiHeadAttention(d_model, num_heads)

    print("end main")


if __name__ == "__main__":
    main()
