import re
import html


def clean_email_text(text):
    """
    Clean email text before passing it to the ML model.
    """

    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
        " ",
        text
    )

    # Keep letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":

    sample_email = """
    URGENT! Your bank account is suspended.

    Click here to verify your password immediately!
    """

    print("Original Email:")
    print(sample_email)

    print("\nCleaned Email:")
    print(clean_email_text(sample_email))