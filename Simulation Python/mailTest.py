import smtplib


#### This is a send mail script to send email to you when simulation is completed. 
# This script is for gmail account

email_address = "Your Email Address"
email_password = "Your Email SMTP Password"

def sendMail(subject,body):
        """
        Send email to yourself 
        """

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(email_address, email_password)
        msg = f"Subject:{subject}\n\n{body}"

        server.sendmail(
                email_address,
                email_address,
                msg
        )
        print('email has been sent')

        server.quit()

def sendMailTo(recipient,subject,body):
        """
        Send email to a recipent.
        
        """
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(email_address, email_password)
        msg = f"Subject:{subject}\n\n{body}"

        server.sendmail(
                email_address,
                recipient,
                msg
        )
        print('email has been sent')

        server.quit()





if __name__ == "__main__":
        sendMail('Test Subject','Body!Body!')
        
    