import base64
import logging
from pathlib import Path
import sendgrid
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType
from config import settings

logger = logging.getLogger(__name__)

def send_report_email(to_email, pdf_path, dataset_name):
    """
    Send an email with the PDF report attached via SendGrid.
    """
    if not settings.SENDGRID_API_KEY:
        logger.warning("SENDGRID_API_KEY is not set. Email will not be sent.")
        return

    sg = sendgrid.SendGridAPIClient(api_key=settings.SENDGRID_API_KEY)
    
    with open(pdf_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
        
    message = Mail(
        from_email='reports@yourdomain.com',
        to_emails=to_email,
        subject=f'Your Scheduled Analysis Report — {dataset_name}',
        html_content=f'<p>Your scheduled analysis for <b>{dataset_name}</b> is attached.</p>'
    )
    
    attachment = Attachment(
        FileContent(encoded),
        FileName('report.pdf'),
        FileType('application/pdf')
    )
    message.attachment = attachment
    
    response = sg.send(message)
    logger.info("Scheduled report email sent successfully to %s. Status code: %s", to_email, response.status_code)
    return response
