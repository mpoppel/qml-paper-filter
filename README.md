# arXiv Daily Paper Monitor

Automated monitoring of arXiv papers related to quantum circuits for machine learning and Fourier series, including papers citing your key references.

## Features

- 🔍 Searches arXiv daily for relevant papers
- 📚 Monitors papers citing your key references (Schuld, McClean, Larocca, etc.)
- 📊 Generates markdown reports with paper summaries
- 📧 **Sends email notifications daily**
- 🤖 Fully automated with GitHub Actions
- 📝 Creates GitHub Issues for easy tracking (optional)
- 🎯 Extended 7-day lookback period (catches delayed publications)

## Quick Setup

### 1. Create Repository

```bash
git clone https://github.com/YOUR_USERNAME/arxiv-monitor.git
cd arxiv-monitor
```

### 2. Create Directory Structure

```bash
mkdir -p .github/workflows reports
touch reports/.gitkeep
```

### 3. Add Files

Copy the provided files:
- `arxiv_monitor.py` → root directory
- `arxiv_monitor.yml` → `.github/workflows/`
- `requirements.txt` → root directory
- `README.md` → root directory

### 4. Configure Email Notifications

#### Option A: Gmail (Recommended)

1. **Create an App Password** (if using 2FA):
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Enable 2-Step Verification if not already enabled
   - Go to "App passwords" or visit: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Name it "GitHub arXiv Monitor"
   - Copy the 16-character password

2. **Add GitHub Secrets**:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Add these secrets:
     - `EMAIL_USERNAME`: Your full Gmail address (e.g., `your.email@gmail.com`)
     - `EMAIL_PASSWORD`: The 16-character app password (NOT your Gmail password)
     - `EMAIL_TO`: Email address to receive notifications (can be same as USERNAME)

#### Option B: Other Email Providers

For Outlook/Office 365:
```yaml
server_address: smtp.office365.com
server_port: 587
```

For Yahoo Mail:
```yaml
server_address: smtp.mail.yahoo.com
server_port: 587
```

Add the same secrets with your provider's credentials.

### 5. Enable Workflow Permissions

1. Go to Settings → Actions → General
2. Under "Workflow permissions":
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

### 6. Commit and Push

```bash
git add .
git commit -m "Initial setup: arXiv paper monitor with email notifications"
git push origin main
```

### 7. Test the Workflow

1. Go to Actions tab
2. Click "arXiv Daily Paper Monitor"
3. Click "Run workflow" → "Run workflow"
4. Check your email in a few minutes!

## Configuration

### Customize Search Queries

Edit `arxiv_monitor.py`:

```python
SEARCH_QUERIES = [
    "quantum circuits AND machine learning",
    "quantum machine learning AND Fourier",
    "variational quantum circuits",
    # Add your custom queries
]

# Monitored papers (from your bibliography)
KEY_REFERENCE_PAPERS = {
    "Schuld 2021": "2008.08605",
    "McClean 2018": "barren plateaus quantum neural network",
    # Add more references
}

DAYS_BACK = 7  # Lookback period in days
```

### Change Schedule

Edit `.github/workflows/arxiv_monitor.yml`:

```yaml
on:
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM UTC
```

**Common schedules:**
- `0 9 * * *` - Daily at 9 AM UTC
- `0 9 * * 1-5` - Weekdays only at 9 AM UTC
- `0 9,17 * * *` - Twice daily (9 AM and 5 PM UTC)
- `0 9 * * MON,WED,FRI` - Monday, Wednesday, Friday at 9 AM UTC

**Time zone conversion:**
- UTC 9:00 = 10:00 CET (winter) / 11:00 CEST (summer)
- UTC 9:00 = 2:00 PST / 3:00 PDT
- UTC 9:00 = 5:00 EST / 6:00 EDT

### Disable Email Notifications

Comment out the email step in the workflow:

```yaml
    # - name: Send email notification
    #   if: always()
    #   uses: dawidd6/action-send-mail@v3
    #   ...
```

### Disable GitHub Issues

Comment out the issue creation step:

```yaml
    # - name: Create GitHub Issue (optional)
    #   ...
```

## Output

### Email Notifications

You'll receive daily emails with:
- Number of new papers found
- Titles and arXiv IDs of papers
- Links to full reports on GitHub
- Attached markdown reports

**Example email:**
```
Subject: 📚 arXiv Digest: 5 new papers (#42)

Found 5 new papers (2025-10-06):

1. Quantum Fourier Analysis for Machine Learning
   https://arxiv.org/abs/2510.xxxxx
   Published: 2025-10-05

2. Barren Plateaus in Deep Quantum Circuits
   ...
```

### GitHub Reports

Daily reports saved in `reports/` directory:
```
reports/
├── arxiv_digest_2025-10-06.md
├── arxiv_digest_2025-10-07.md
└── ...
```

### GitHub Issues (Optional)

Issues created daily with full reports and tagged with `arxiv`, `automated`, `papers`.

## Key Reference Papers Monitored

The system automatically searches for papers citing:

1. **Schuld et al. 2021** - Effect of data encoding on expressive power
2. **McClean et al. 2018** - Barren plateaus in quantum neural networks
3. **Pérez-Salinas et al. 2020** - Data re-uploading for universal quantum classifier
4. **Larocca et al. 2024** - Review of barren plateaus
5. **Abbas et al. 2021** - Power of quantum neural networks
6. **Cerezo et al. 2021** - Variational quantum algorithms
7. **Mitarai et al. 2018** - Quantum circuit learning
8. **Caro et al. 2022** - Generalization in quantum machine learning
9. **Ragone et al. 2024** - Lie algebraic theory of barren plateaus
10. **Barthe et al. 2024** - Gradients and frequency profiles

Add more references by editing `KEY_REFERENCE_PAPERS` in `arxiv_monitor.py`.

## Troubleshooting

### Email Not Sending

**Check secrets:**
```bash
# Secrets should be set:
EMAIL_USERNAME=your.email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_TO=recipient@email.com
```

**Gmail issues:**
- Use App Password, not Gmail password
- Ensure 2FA is enabled
- App password should be 16 characters (no spaces)

**Test locally:**
```bash
pip install arxiv
python arxiv_monitor.py
# Check reports/email_body.txt
```

### No Papers Found

- Papers aren't published every day
- Extend `DAYS_BACK` to 7-14 days initially
- Check arXiv manually: https://arxiv.org/list/quant-ph/recent
- Broaden search queries

### Workflow Not Running

- Check Actions → Enable workflows if disabled
- Verify cron syntax at [crontab.guru](https://crontab.guru)
- Manual trigger: Actions → Run workflow
- Check workflow permissions in Settings

### Permission Errors

Settings → Actions → General → Workflow permissions:
- Select "Read and write permissions"
- Enable "Allow GitHub Actions to create and approve pull requests"

## Local Testing

Test without email:

```bash
pip install -r requirements.txt
python arxiv_monitor.py

# Check outputs
cat reports/email_body.txt
ls -la reports/
```

## Advanced Features

### Add More Email Recipients

Edit workflow to send to multiple addresses:

```yaml
to: user1@email.com,user2@email.com,user3@email.com
```

### Customize Email Format

Edit the `html_body` section in the workflow for HTML emails.

### Add Slack Notifications

Add this step after email:

```yaml
- name: Send Slack notification
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    text: 'Found ${{ steps.search.outputs.paper_count }} new papers!'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Filter by Author

Add to `arxiv_monitor.py`:

```python
FAVORITE_AUTHORS = ["Schuld", "Cerezo", "McClean"]

if any(author in fav_author for author in paper.authors 
       for fav_author in FAVORITE_AUTHORS):
    # Highlight or prioritize
```

## Resources

- [arXiv API](https://info.arxiv.org/help/api/index.html)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)
- [Cron Expression Reference](https://crontab.guru/)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review GitHub Actions logs
3. Test locally with Python
4. Check email provider settings

## License

MIT License - customize freely for your research!