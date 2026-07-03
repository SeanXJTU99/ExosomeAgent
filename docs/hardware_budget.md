# Hardware Budget

## Target Configuration

Single consumer GPU deployment for private, on-premises exosome CRO customer service.

| Component | Specification | Est. Cost (CNY) |
|-----------|--------------|------------------|
| GPU | NVIDIA RTX 4090 24G (used) | ~8,000 |
| CPU | Intel i5-13600KF or equivalent | ~1,500 |
| Motherboard | Z790 / B760 (PCIe 4.0 x16) | ~1,200 |
| RAM | DDR5 32 GB (2 × 16 GB) | ~600 |
| Storage | 1 TB NVMe SSD (PCIe 4.0) | ~400 |
| PSU | 850W 80+ Gold | ~600 |
| Case + Cooling | ATX mid-tower + air cooling | ~400 |
| **Total** | | **~12,700** |

**Budget**: < ¥15,000 (target met with ~15% margin)

## Alternative: RTX 3090 24G

| Component | Specification | Est. Cost (CNY) |
|-----------|--------------|------------------|
| GPU | NVIDIA RTX 3090 24G (used) | ~5,500 |
| Other components | Same as above | ~4,700 |
| **Total** | | **~10,200** |

RTX 3090 has the same 24 GB VRAM but slower tensor cores. Still sufficient for Qwen2.5-14B-AWQ-INT4 with 4 concurrent requests.

## Alternative: Qwen2.5-7B (Lower Cost)

| Change | Detail |
|--------|--------|
| Model | Qwen2.5-7B-Instruct AWQ INT4 (~3.8 GB) |
| GPU option | RTX 4060 Ti 16G (~3,200 CNY) |
| Total system cost | ~7,000 CNY |

Trade-off: 7B model has lower domain accuracy. Recommended only if budget is severely constrained.

## VRAM Breakdown (14B Model)

```
RTX 4090 24G Total VRAM
├── Model weights (INT4):      7.5 GB  (31%)
├── KV Cache (8K tokens):      6.3 GB  (26%)
├── vLLM runtime:              2.0 GB  ( 8%)
├── LangGraph + FAISS:         1.0 GB  ( 4%)
├── CUDA context + overhead:   1.0 GB  ( 4%)
├── Headroom:                  6.2 GB  (26%)
└── Total:                    24.0 GB
```

## Not Included

The ¥15,000 budget covers only the inference server. Model training (QLoRA + DPO) requires the same GPU but more time — approximately 6–8 hours for the full three-stage curriculum on a single RTX 4090. No additional hardware is needed for training.
