# Quantum ML Paper Filter

Automated filtering of arXiv papers for quantum machine learning research, focusing on data encoding and variational quantum circuits.

## 🎯 What It Does

- **Daily monitoring** of arXiv papers across quantum physics and machine learning categories
- **Smart filtering** using keywords, author recognition, and reference analysis
- **Reference tracking** for papers citing your seminal work (arXiv:2008.08605)
- **Email digests** with the most relevant papers delivered daily
- **Zero maintenance** - runs automatically on GitHub Actions

## 🚀 Quick Start

1. **Fork/clone this repository**
2. **Set up Gmail app password** (see setup guide below)
3. **Configure GitHub Secrets** with your email credentials
4. **Customize keywords** in `src/config.py`
5. **Enable GitHub Actions** and you're done!

## ⚙️ Setup Guide

### Gmail Configuration
1. Enable 2-factor authentication on your Gmail account
2. Go to Google Account → Security → 2-Step Verification → App passwords
3. Generate an app password for "Mail"
4. Save this password for the next step

### GitHub Secrets
Add these secrets in your repository settings (Settings → Secrets and variables → Actions):

**Required:**
- `EMAIL_USERNAME`: your-email@gmail.com
- `EMAIL_PASSWORD`: your-gmail-app-password
- `EMAIL_TO`: recipient@example.com

**Optional:**
- `SLACK_WEBHOOK_URL`: https://hooks.slack.com/services/...

### Customization
Edit `src/config.py` to:
- Add your specific research keywords
- Include authors you want to track
- Add more seminal papers relevant to your work
- Adjust scoring weights

## 📊 Features

- **Multi-layered filtering**: Keywords, authors, categories, references
- **Reference analysis**: Uses Semantic Scholar API to find citation relationships
- **Smart scoring**: Weighted relevance based on paper characteristics
- **Rich outputs**: HTML emails, JSON data, plain text summaries
- **Error handling**: Automatic issue creation and notification on failures
- **Rate limiting**: Respects API limits and implements delays

## 🔧 How It Works

1. **Fetches** recent papers from arXiv (quant-ph, cs.LG, cs.AI, etc.)
2. **Scores** papers based on:
   - Keyword matches in titles and abstracts
   - Author recognition (Maria Schuld, Seth Lloyd, etc.)
   - References to seminal papers
   - Heuristic signals for methodological similarity
3. **Ranks** papers by combined relevance score
4. **Generates** digest with top papers
5. **Emails** results as formatted HTML with links

## 📈 Sample Output

```
🔬 Quantum ML Daily Digest - 2025-09-04
Found 8 relevant papers

1. Enhanced Data Encoding Strategies for Variational Quantum Classifiers
   arXiv:2409.12345 | Combined Score: 34.5
   Authors: Smith, J. et al.
   Relevance: Primary: quantum machine learning; Encoding: data encoding
   References: 12 points
```

## 🔄 Schedule

- **Daily**: 9 AM UTC (configurable in workflow file)
- **Manual**: Can be triggered anytime from Actions tab
- **Weekly**: Optional summary (currently disabled)

## 🛠 Local Testing

```bash
cd src
pip install -r ../requirements.txt
python qml_filter.py --days-back 1
```

## 📝 License

MIT License - feel free to fork and customize for your research area!

## 🤝 Contributing

Issues and pull requests welcome! This tool can easily be adapted for other research domains.
