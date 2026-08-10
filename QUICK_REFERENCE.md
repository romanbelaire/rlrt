# Topic-Level Evaluation - Quick Reference Card

## 🚀 Fast Start

### Pilot Test (10-30 min)
```bash
# Submit SLURM job for pilot test
sbatch scripts/run_pilot_topic_evaluation.sh outputs/MODEL/run_DATE/final_model

# Monitor job
squeue -u $USER
tail -f rbelaire.2021.topic-pilot.JOBID.out
```

### Full Evaluation (6-8 hours)
```bash
# Submit SLURM job for full evaluation
sbatch scripts/run_topic_level_evaluation.sh outputs/MODEL/run_DATE/final_model

# Monitor job
squeue -u $USER
tail -f rbelaire.2021.topic-eval.JOBID.out
```

---

## 📊 Topics (7 Categories)

| Topic | Count | % |
|-------|-------|---|
| copyright | 100 | 25% |
| cybercrime_intrusion | 67 | 17% |
| illegal | 65 | 16% |
| misinformation_disinformation | 65 | 16% |
| chemical_biological | 56 | 14% |
| harassment_bullying | 25 | 6% |
| harmful | 22 | 6% |

---

## 🎯 Common Commands

### Specific Topics
```bash
# Set environment variable before submitting
export SPECIFIC_TOPICS="copyright illegal cybercrime_intrusion"
sbatch scripts/run_topic_level_evaluation.sh MODEL
```

### Fast Mode (Fewer Episodes)
```bash
# Set environment variables for quick run
export EPISODES_PER_TOPIC=10
export MAX_QUERIES=20
sbatch scripts/run_topic_level_evaluation.sh MODEL
```

### Memory Constrained
```bash
# Use smaller batch size
export BATCH_SIZE=2
sbatch scripts/run_topic_level_evaluation.sh MODEL
```

### Skip Auto-Analysis
```bash
# Skip analysis step
export SKIP_ANALYSIS=true
sbatch scripts/run_topic_level_evaluation.sh MODEL
```

---

## 📁 Output Structure

```
topic_level_results/MODEL_NAME/
├── summary.json                    # All metrics
├── results_by_topic.csv           # Table view
├── analysis_report.md             # Report
├── visualizations/*.pdf           # Plots
└── TOPIC_NAME/                    # Logs
    ├── conversations.jsonl
    └── metrics.json
```

---

## 📈 Key Metrics

- **ASR**: Attack Success Rate (harmful outputs / total)
- **Harm Score**: Guard model score (0-1, higher = more harmful)
- **Conv Length**: Average turns to success
- **First Turn Success**: Attacks successful immediately

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of Memory | `--batch-size 2` |
| Too Slow | `--episodes 10 --max-queries 20` |
| Missing Deps | `pip install pandas numpy matplotlib seaborn scipy` |

---

## 📖 Documentation

- **Quick Start**: `docs/TOPIC_LEVEL_EVALUATION_README.md`
- **Full Plan**: `docs/TOPIC_LEVEL_EVALUATION_PLAN.md`
- **Summary**: `TOPIC_EVALUATION_SUMMARY.md`

---

## 🎓 Example Workflow

1. **Test**: `sbatch scripts/run_pilot_topic_evaluation.sh MODEL`
2. **Monitor**: `tail -f rbelaire.2021.topic-pilot.*.out`
3. **Review**: `cat topic_level_results/pilot_*/analysis_report.md`
4. **Full Run**: `sbatch scripts/run_topic_level_evaluation.sh MODEL`
5. **Analyze**: Results auto-generated in `topic_level_results/`

---

## ⏱️ Time Estimates

| Mode | Episodes | Topics | Time |
|------|----------|--------|------|
| Pilot | 5 | 2 | 10-15 min |
| Quick | 10 | 7 | 2-3 hours |
| Standard | 20 | 7 | 6-8 hours |
| Thorough | 50 | 7 | 15-20 hours |

---

## 🖥️ GPU Requirements

- **Minimum**: 40GB GPU (1x A40)
- **Optimal**: 48GB+ GPU (A100)
- **Best**: 2x GPUs (separate models)

---

## ✅ Available Models

### Primary
```
outputs/normal-agent-8b8t-std/run_20250801_195707/final_model
```

### Ablations
```
outputs/hl_only/run_20250821_105532/final_model
outputs/template-ablation-*/run_*/final_model
```

### Variants
```
outputs/normal-agent-8b8t-uniform/run_*/final_model
outputs/normal-agent-8b8t-no-pairwise/run_*/final_model
```

---

## 📊 Interpreting Results

### ANOVA
- **p < 0.05**: Topics differ significantly
- **p ≥ 0.05**: Consistent across topics

### Success Rate Ranking
- **High ASR (>0.7)**: Easy to attack
- **Medium ASR (0.4-0.7)**: Moderate difficulty
- **Low ASR (<0.4)**: Hard to attack

### Harm Scores
- **>0.8**: Very harmful outputs
- **0.5-0.8**: Moderately harmful
- **<0.5**: Marginally harmful

---

**Ready to start? Run the pilot test!**

```bash
# Submit pilot test job
sbatch scripts/run_pilot_topic_evaluation.sh \
    outputs/normal-agent-8b8t-std/run_20250801_195707/final_model

# Check job status
squeue -u $USER

# Monitor output (replace JOBID with actual job ID)
tail -f rbelaire.2021.topic-pilot.JOBID.out
```

