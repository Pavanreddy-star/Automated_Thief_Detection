import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

def send_email_alert(image_path):
    sender_email = "ppavankumarreddy3202@gmail.com"
    receiver_email = "ppavankumarreddy1234@gmail.com"
    password = "icxp qxde hbnq agsi"  # or use the generated app-specific password
    subject = "Thief Detected!"
    body = "A thief has been detected. Please find the attached image."

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    try:
        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            msg.attach(img)
    except Exception as e:
        print(f"Error reading image: {e}")
        return  # Return early if there's an issue with the image

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)  # For SSL
        # server = smtplib.SMTP("smtp.gmail.com", 587)  # For TLS
        # server.starttls()  # Uncomment if using TLS instead of SSL

        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Alert sent successfully!")
    except Exception as e:
        print(f"Error: {e}")