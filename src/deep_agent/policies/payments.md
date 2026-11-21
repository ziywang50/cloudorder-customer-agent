# Payments: charges versus holds

- An AUTHORIZATION reserves funds on the customer's card. It is not a charge.
- A CAPTURE moves money. Only successful captures count as charges.
- A VOIDED authorization was cancelled by CloudOrder. The customer's bank may keep showing it
  as "pending" until the hold is released, usually within 5 days. No refund is needed or
  possible. Explain this and give the expected release date (holdReleaseExpectedBy) when
  available.
- When a customer reports "two charges", compare successful captures with the order total
  before concluding anything.
