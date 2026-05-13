import torch
import ttnn


def main():
    device = None
    try:
        print("Opening TT device...")
        device = ttnn.open_device(device_id=0)

        print("Creating torch tensors...")
        a_torch = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
        b_torch = torch.full((1, 1, 32, 32), 2.0, dtype=torch.bfloat16)

        print("Moving tensors to TT device...")
        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        print("Running TTNN add...")
        c = ttnn.add(a, b)

        print("Moving result back to torch...")
        c_torch = ttnn.to_torch(c)

        print("Result shape:", c_torch.shape)
        print("Result[0,0,0,0]:", c_torch[0, 0, 0, 0].item())

        expected = torch.full((1, 1, 32, 32), 3.0, dtype=torch.bfloat16)
        ok = torch.allclose(c_torch, expected)

        print("PASS" if ok else "FAIL")

    finally:
        if device is not None:
            print("Closing TT device...")
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
