"""Build mock help center articles for Acme Store from Bitext dataset.

Reads data/ training split via pipeline.ingest, extracts intent metadata,
and generates clean, deterministic markdown help-center articles for all 27 intents.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest import DEFAULT_DATA_FILE, prepare_dataset

KB_DIR = PROJECT_ROOT / "kb"

ARTICLES = {
    "cancel_order": {
        "title": "How to Cancel an Order",
        "summary": "Learn how to cancel an order placed with Acme Store before it enters warehouse fulfillment. Once an order has shipped, cancellation is no longer possible online and the package must be returned instead.",
        "steps": [
            "Log in to your Acme Store account at acme-store.example.com.",
            "Navigate to 'Order History' under your Account dashboard.",
            "Locate the order you wish to cancel using your order number (#ACME-12345) and click 'Order Details'.",
            "Select the 'Cancel Order' option and indicate a reason for the cancellation.",
            "Confirm the cancellation request and check your email inbox for an immediate cancellation receipt.",
        ],
        "faqs": [
            (
                "Can I cancel an order after it has shipped?",
                "Once an order has shipped from our warehouse, it cannot be canceled directly. You can refuse delivery upon arrival or initiate a return within 30 days of delivery for a full refund.",
            ),
            (
                "When will my refund be processed after canceling an order?",
                "If your order is canceled before fulfillment, any payment hold or charge is refunded immediately. Funds typically appear on your original payment method within 3 to 5 business days.",
            ),
        ],
    },
    "change_order": {
        "title": "How to Modify an Existing Order",
        "summary": "Modify item quantities, product variations, or remove items from a pending order before our warehouse begins fulfillment. Once an order is being packed, modifications cannot be made.",
        "steps": [
            "Sign in to your account at acme-store.example.com and open 'Order History'.",
            "Select the pending order you wish to adjust and click 'View Details'.",
            "Click 'Modify Order' next to the items you want to change.",
            "Adjust quantities, remove unwanted items, or select alternative sizes or colors.",
            "Review your updated cart total and click 'Save Changes' to confirm the modifications.",
        ],
        "faqs": [
            (
                "Can I add new items to an existing order?",
                "You can add items as long as the order status remains 'Pending'. If the order has already moved to 'Processing', please place a separate new order on acme-store.example.com.",
            ),
            (
                "Why is the 'Modify Order' button missing?",
                "If the modification button is unavailable, your order has already been sent to our fulfillment center for packing and can no longer be edited.",
            ),
        ],
    },
    "change_shipping_address": {
        "title": "How to Update Your Shipping Address",
        "summary": "Update the delivery destination for an active order before dispatch or change your default shipping address for future Acme Store purchases.",
        "steps": [
            "Log in to your account at acme-store.example.com.",
            "Go to 'Order History' and select the pending order requiring an address update.",
            "Locate the 'Shipping Address' section on the order summary page and click 'Edit Address'.",
            "Enter your updated street address, apartment or suite number, city, state, and zip code.",
            "Click 'Update Address' to save the new delivery destination for your shipment.",
        ],
        "faqs": [
            (
                "Can I change the delivery address after an order ships?",
                "Once a carrier tracking number has been generated, address corrections cannot be made through our site. Contact the carrier directly (UPS, FedEx, or USPS) to request package rerouting.",
            ),
            (
                "Can I change the destination country on a pending order?",
                "Destination countries cannot be changed on existing orders due to tax and shipping recalculations. Please cancel the current order and place a new one with the international address.",
            ),
        ],
    },
    "check_cancellation_fee": {
        "title": "Cancellation Fees and Policy",
        "summary": "Acme Store maintains a customer-friendly cancellation policy. Understand when cancellations are free and the rare circumstances where restocking or processing fees apply.",
        "steps": [
            "Visit acme-store.example.com and navigate to the 'Help Center' or 'Account Terms'.",
            "Select 'Cancellation Policy' under the Order Management category.",
            "Verify the classification of your items (standard consumer goods vs. custom-manufactured merchandise).",
            "Check your order status under 'Order History' to confirm eligibility for no-fee cancellation.",
        ],
        "faqs": [
            (
                "Does Acme Store charge fees for canceling standard orders?",
                "No. There are never any fees or penalties for canceling standard orders prior to warehouse dispatch and shipment.",
            ),
            (
                "Do customized or special-order products carry a cancellation fee?",
                "Customized or engraved items canceled after production has begun may be subject to a 15% material restocking fee as disclosed at checkout.",
            ),
        ],
    },
    "check_invoice": {
        "title": "How to View and Inspect Invoices",
        "summary": "Access, review, and verify detailed invoice statements, sales tax breakdowns, and itemized receipts for all your Acme Store purchases.",
        "steps": [
            "Log in to your Acme Store account at acme-store.example.com.",
            "Open your account menu in the top-right corner and select 'Billing & Invoices'.",
            "Filter records by date range or enter your invoice number (e.g., #INV-12345).",
            "Click 'View Details' on the selected invoice to inspect item prices, discounts, and applied taxes.",
        ],
        "faqs": [
            (
                "How long are past invoices accessible in my account?",
                "All digital invoices and purchase records remain accessible in your Acme Store account history for up to 7 years.",
            ),
            (
                "What should I do if an invoice contains incorrect billing information?",
                "Contact our billing department at billing@acme-store.example.com with your invoice number (#INV-12345) and requested corrections for a revised statement.",
            ),
        ],
    },
    "check_payment_methods": {
        "title": "Accepted Payment Methods",
        "summary": "Acme Store supports a wide selection of convenient, secure payment methods for purchases made through our website, mobile platform, and retail locations.",
        "steps": [
            "Add desired merchandise to your shopping cart and proceed to Checkout at acme-store.example.com.",
            "At the 'Payment Method' step, view the available payment options.",
            "Choose your preferred payment method: Major Credit/Debit Cards (Visa, MasterCard, American Express, Discover), PayPal, Apple Pay, Google Pay, or Acme Store Gift Cards.",
            "Enter your card details, account credentials, or gift card claim code.",
            "Optionally select 'Save Payment Method' to safely store credentials for one-click checkout on future orders.",
        ],
        "faqs": [
            (
                "Can I split payment across multiple credit cards?",
                "Our checkout system supports split payments only between an Acme Store Gift Card and one secondary payment method such as a credit card or PayPal.",
            ),
            (
                "Are my credit card and banking details secure?",
                "Yes. Acme Store uses bank-level 256-bit SSL encryption and strict PCI-DSS Level 1 compliance to ensure your payment details are never compromised or stored in plain text.",
            ),
        ],
    },
    "check_refund_policy": {
        "title": "Refund and Return Policy",
        "summary": "Review Acme Store's 30-day money-back guarantee, return requirements, inspection procedures, and payout conditions for online and in-store purchases.",
        "steps": [
            "Visit acme-store.example.com/refunds to review full policy details and terms.",
            "Verify that your item was delivered within the last 30 calendar days.",
            "Ensure returned merchandise is in original, undamaged condition with all accessories, manuals, and original packaging intact.",
            "Initiate your return through 'Returns & Refunds' in your account dashboard to obtain a prepaid return shipping label.",
        ],
        "faqs": [
            (
                "What situations qualify for an immediate 100% refund?",
                "Defective products, items damaged during shipping, incorrect merchandise sent, or orders canceled before shipment qualify for a complete refund including original shipping charges.",
            ),
            (
                "How long does it take for a refund to post to my bank account?",
                "Once your returned item arrives at our facility and passes inspection (typically 1 to 2 business days), refunds are issued immediately. Banks post funds within 3 to 5 business days.",
            ),
        ],
    },
    "complaint": {
        "title": "How to File a Customer Complaint",
        "summary": "We take service quality and customer satisfaction seriously. Learn how to lodge a formal complaint and escalate issues to our dedicated dispute resolution team.",
        "steps": [
            "Gather all pertinent information, including your order number (#ACME-12345), dates of interaction, photos of damaged goods, and a brief description of the issue.",
            "Visit the complaint portal at acme-store.example.com/support/complaints or email complaints@acme-store.example.com.",
            "Fill out the dispute intake form with your contact details and attach any supporting documentation.",
            "Submit the complaint to receive a dedicated case number (#CASE-54321) within 2 hours.",
            "A senior resolution specialist will investigate your case and provide an official response within 1 business day.",
        ],
        "faqs": [
            (
                "Can I escalate my complaint if I am unhappy with the initial response?",
                "Yes. You can reply directly to your case manager requesting supervisor review, or call 1-800-555-ACME and ask for the Customer Advocacy Team.",
            ),
            (
                "Is there an external arbitration option for unresolved disputes?",
                "If a resolution cannot be reached through our claims team, you may request formal mediation in accordance with Acme Store terms of service.",
            ),
        ],
    },
    "contact_customer_service": {
        "title": "How to Contact Acme Store Customer Support",
        "summary": "Find all official contact channels, operating hours, phone numbers, and email addresses to reach Acme Store customer care.",
        "steps": [
            "Review our support channels to choose the fastest option for your inquiry: Phone, Live Chat, or Email.",
            "For Phone Support, call toll-free at 1-800-555-ACME (1-800-555-2263) during customer service hours: Monday through Friday, 8:00 AM – 8:00 PM EST.",
            "For Live Chat, visit acme-store.example.com and click the chat icon in the bottom-right corner for instant messaging.",
            "For Email Support, send your detailed query to support@acme-store.example.com for a guaranteed response within 24 business hours.",
        ],
        "faqs": [
            (
                "Is phone customer service completely free?",
                "Yes. 1-800-555-ACME is a toll-free number accessible throughout the United States and Canada.",
            ),
            (
                "How can I get help outside of standard business hours?",
                "Outside of operating hours (8:00 AM – 8:00 PM EST), you can submit an email ticket at support@acme-store.example.com or browse self-service articles at acme-store.example.com/help.",
            ),
        ],
    },
    "contact_human_agent": {
        "title": "How to Speak with a Live Human Agent",
        "summary": "Connect directly with a live customer care representative when automated help articles and virtual assistants are insufficient.",
        "steps": [
            "Visit acme-store.example.com and launch the Live Chat widget in the bottom corner of your screen.",
            "In the chat prompt, type 'Speak with agent' or select 'Connect to Human' from the assistance menu.",
            "Remain in the chat window while our system queues the next available support specialist (typical wait < 2 minutes).",
            "Alternatively, dial 1-800-555-ACME on your phone, press 0, and say 'Customer Service Representative' at the voice menu.",
        ],
        "faqs": [
            (
                "What are the average wait times to reach a human agent?",
                "Average queue wait times are under 2 minutes for online live chat and between 3 to 5 minutes for telephone support during normal operating hours.",
            ),
            (
                "Can an agent call me back rather than having me wait on hold?",
                "Yes. If telephone queue hold times exceed 5 minutes, our automated phone system will offer you a callback option while retaining your priority in line.",
            ),
        ],
    },
    "create_account": {
        "title": "How to Create an Acme Store Account",
        "summary": "Set up a free Acme Store customer account to enjoy expedited checkout, shipment tracking, order history access, and members-only discounts.",
        "steps": [
            "Visit acme-store.example.com and click 'Register' or 'Sign In' in the top header navigation.",
            "Click the 'Create an Account' button on the sign-in modal.",
            "Enter your full name, a valid email address, and choose a secure password (minimum 8 characters).",
            "Review and accept the Acme Store Terms of Service and Privacy Policy.",
            "Click 'Create Account' and click the activation link sent to your email to verify your address.",
        ],
        "faqs": [
            (
                "Are there any fees or subscriptions required to create an account?",
                "No. Creating an account at Acme Store is completely free with no recurring fees or maintenance charges.",
            ),
            (
                "Can I sign up using Google or Apple single sign-on?",
                "Yes. You can click 'Continue with Google' or 'Continue with Apple' on the registration screen for instant account setup.",
            ),
        ],
    },
    "delete_account": {
        "title": "How to Delete or Close Your Account",
        "summary": "Permanently close your Acme Store account and remove your personal information in compliance with privacy regulations.",
        "steps": [
            "Log in to your account at acme-store.example.com using your current credentials.",
            "Click on your profile icon and select 'Account Settings'.",
            "Navigate to the 'Privacy & Security' tab and scroll down to 'Account Closure'.",
            "Click 'Delete Account' and review the notice regarding the loss of order records and rewards.",
            "Enter your password to verify your identity and click 'Confirm Permanent Deletion'.",
        ],
        "faqs": [
            (
                "Can an account be restored after it has been deleted?",
                "No. Account deletion is permanent and irreversible. All saved addresses, stored payment methods, and historical order receipts will be deleted.",
            ),
            (
                "Can I delete my account if I have an active order or pending return?",
                "No. Accounts cannot be closed while there are in-flight orders, pending deliveries, or active return refunds. Please wait until all transactions settle.",
            ),
        ],
    },
    "delivery_options": {
        "title": "Delivery Methods and Shipping Speeds",
        "summary": "Explore our shipping options, delivery speeds, carrier partners, and rates for orders shipped across the United States and internationally.",
        "steps": [
            "Add items to your cart and proceed to the checkout screen at acme-store.example.com.",
            "Enter your destination shipping address to display available delivery methods and estimated arrival dates.",
            "Choose 'Standard Shipping' (3 to 5 business days, free on orders over $50).",
            "Choose 'Expedited Shipping' (2 business days) or 'Overnight Express' (next business day delivery) for time-sensitive orders.",
            "Choose 'In-Store Pickup' if you prefer to collect your items for free at a local Acme Store retail location.",
        ],
        "faqs": [
            (
                "Does Acme Store offer free shipping?",
                "Yes. Standard shipping is completely free on all domestic orders totaling $50 or more before taxes.",
            ),
            (
                "What carriers does Acme Store use for package delivery?",
                "We partner with major carriers including UPS, FedEx, and the United States Postal Service (USPS) depending on the shipping method selected and package weight.",
            ),
        ],
    },
    "delivery_period": {
        "title": "Estimated Delivery Times and Transit Windows",
        "summary": "Understand delivery timeframes, tracking updates, and when to expect package arrival following dispatch from our fulfillment centers.",
        "steps": [
            "Review your order confirmation email for the estimated delivery window calculated at time of purchase.",
            "Once your order dispatches, locate your tracking number (e.g., ACME-TRK-98765) in the shipping confirmation email.",
            "Visit acme-store.example.com/track or the carrier's tracking page to see real-time transit scans.",
            "Note our standard transit windows: Standard (3-5 business days), Expedited (2 business days), Overnight (1 business day).",
        ],
        "faqs": [
            (
                "What is the daily cutoff time for same-day shipping?",
                "Orders placed Monday through Friday before 2:00 PM EST ship the same day. Orders placed after 2:00 PM EST ship the following business day.",
            ),
            (
                "What should I do if my package has not arrived past the estimated date?",
                "If your shipment is delayed more than 48 hours beyond the estimated delivery date, contact support@acme-store.example.com with your tracking number for an immediate carrier inquiry.",
            ),
        ],
    },
    "edit_account": {
        "title": "How to Edit Your Account Information",
        "summary": "Keep your personal profile, email address, phone number, and communication preferences current in your Acme Store account settings.",
        "steps": [
            "Sign in to your Acme Store account at acme-store.example.com.",
            "Click on your profile avatar in the upper-right corner and select 'Account Settings'.",
            "Under the 'Personal Details' section, click 'Edit' next to your name, contact phone number, or email.",
            "Update the relevant information in the form fields.",
            "Click 'Save Changes' to apply your updates. If you changed your email address, confirm the verification link sent to your new address.",
        ],
        "faqs": [
            (
                "Will editing my account profile update existing pending orders?",
                "No. Updating account details only affects future transactions. To update a shipping address on an open order, modify the order directly under 'Order History'.",
            ),
            (
                "Why was I prompted to re-enter my password when saving changes?",
                "To protect your security, changes to sensitive profile details like email address or password require re-entering your current password.",
            ),
        ],
    },
    "get_invoice": {
        "title": "How to Download or Print an Invoice",
        "summary": "Obtain official PDF copies of invoices, receipts, and order summaries for accounting, tax reporting, or expense reimbursement.",
        "steps": [
            "Sign in to your Acme Store account at acme-store.example.com.",
            "Navigate to 'Order History' or 'Billing & Invoices' from your account menu.",
            "Find the order corresponding to the desired invoice using your order number (#ACME-12345).",
            "Click the 'Download Invoice (PDF)' or 'Print Receipt' link next to the order summary.",
            "Save the downloaded PDF to your local device or print a physical copy for your records.",
        ],
        "faqs": [
            (
                "Can I obtain an invoice with company VAT or corporate tax details?",
                "Yes. Add your business name and Tax/VAT ID in 'Account Settings' under 'Company Billing' before generating the invoice download.",
            ),
            (
                "Can customer service email an invoice directly to my accountant?",
                "Yes. Send a request to billing@acme-store.example.com mentioning your order number (#ACME-12345) and the recipient email address.",
            ),
        ],
    },
    "get_refund": {
        "title": "How to Request a Refund",
        "summary": "Step-by-step guidance on initiating product returns, reporting non-receipt or defective goods, and receiving money back.",
        "steps": [
            "Log in to your account at acme-store.example.com and visit the 'Returns & Refunds' portal.",
            "Select the order containing the items you wish to return.",
            "Choose a return reason (e.g., defective merchandise, item not as described, or changed mind) and select refund to original payment method.",
            "Download and print the prepaid return shipping label provided by Acme Store.",
            "Pack the item securely in its original packaging, affix the label, and drop off the package at an authorized carrier location.",
        ],
        "faqs": [
            (
                "Are return shipping labels free for customer returns?",
                "Return shipping is 100% free for defective, damaged, or incorrectly shipped items. Standard preference returns have a flat $4.99 fee deducted from the final refund.",
            ),
            (
                "How will the refund be issued?",
                "All refunds are automatically credited back to the original payment method used during checkout (credit card, PayPal, or gift card).",
            ),
        ],
    },
    "newsletter_subscription": {
        "title": "Managing Your Newsletter Subscription",
        "summary": "Subscribe to receive promotional coupons, weekly deals, and product release updates, or unsubscribe from marketing emails at any time.",
        "steps": [
            "To Subscribe: Scroll to the footer of acme-store.example.com, enter your email address into 'Stay Connected', and click 'Subscribe'.",
            "To Unsubscribe via Email: Open any marketing email from Acme Store and click the 'Unsubscribe' link located at the bottom of the message.",
            "To Unsubscribe via Account: Log in at acme-store.example.com, go to 'Account Settings' > 'Email Preferences', and uncheck 'Promotional Newsletters'.",
            "Click 'Save Preferences' to update your communication settings.",
        ],
        "faqs": [
            (
                "Will unsubscribing from newsletters cancel my order notifications?",
                "No. Unsubscribing from marketing emails will never suppress critical transactional messages such as order confirmations, tracking alerts, or security notices.",
            ),
            (
                "How quickly does an unsubscribe request take effect?",
                "Unsubscribe requests are applied immediately in our system, though please allow up to 24 hours for scheduled marketing campaigns to fully cease.",
            ),
        ],
    },
    "payment_issue": {
        "title": "Troubleshooting Payment and Checkout Errors",
        "summary": "Resolve card declines, authorization failures, billing address mismatches, and checkout errors when shopping at Acme Store.",
        "steps": [
            "Check card details: Confirm that the card number, expiration date, and 3- or 4-digit CVV security code were typed correctly without spaces.",
            "Confirm billing address: Make sure the billing street address and zip code match the exact billing address on file with your bank or credit card company.",
            "Verify funds and card limits: Ensure your account has sufficient available credit or balance and that online transactions are permitted by your card issuer.",
            "Try an alternative payment method: Switch to an alternative option such as PayPal, Apple Pay, Google Pay, or a different credit card.",
            "Contact our billing team: If issues persist, call 1-800-555-ACME or email support@acme-store.example.com for checkout assistance.",
        ],
        "faqs": [
            (
                "Why do I see a charge on my bank statement if my order failed?",
                "A temporary authorization hold may appear on your bank statement following a failed attempt. This is not an actual charge and will automatically disappear within 24 to 48 hours.",
            ),
            (
                "What should I do if my payment was declined due to fraud protection?",
                "Contact your bank or card provider to authorize the transaction from Acme Store, then retry completing your order.",
            ),
        ],
    },
    "place_order": {
        "title": "How to Place an Order on Acme Store",
        "summary": "A comprehensive guide on searching our product catalog, adding items to your shopping cart, and checking out smoothly.",
        "steps": [
            "Browse catalog categories or enter keywords into the search bar at acme-store.example.com.",
            "Click on a product to view item details, specifications, available sizes, colors, and pricing.",
            "Select your options and click 'Add to Cart', then click 'View Cart & Checkout' in the cart slide-out.",
            "Enter your shipping address, choose your desired delivery speed, and review your order summary.",
            "Enter payment details, apply any promotional discount codes, and click 'Place Your Order'.",
        ],
        "faqs": [
            (
                "Can I complete an order as a guest without creating an account?",
                "Yes. Acme Store supports guest checkout. You only need to provide an email address for order confirmation and shipping updates.",
            ),
            (
                "Where do I enter a promo code or gift card during checkout?",
                "On the Checkout Payment page, locate the 'Gift Card or Promo Code' field, enter your code, and click 'Apply' to see your revised total.",
            ),
        ],
    },
    "recover_password": {
        "title": "How to Recover or Reset Your Password",
        "summary": "Reset a forgotten password or access credential to regain access to your Acme Store account securely.",
        "steps": [
            "Go to the Acme Store login screen at acme-store.example.com/login.",
            "Click the 'Forgot Password?' link located directly beneath the password input box.",
            "Enter the email address registered with your Acme Store account and click 'Send Password Reset Link'.",
            "Check your inbox for an email from support@acme-store.example.com titled 'Reset Your Acme Store Password'.",
            "Click the secure reset link inside the email, enter your new password, confirm it, and click 'Save New Password'.",
        ],
        "faqs": [
            (
                "What should I do if the password reset email does not arrive?",
                "Check your spam or junk mail folders. If the email is not found within 5 minutes, ensure the email entered exactly matches your registered account.",
            ),
            (
                "How long does the password reset link stay active?",
                "For account security, password reset links remain valid for 60 minutes from generation. After 60 minutes, a new reset request must be submitted.",
            ),
        ],
    },
    "registration_problems": {
        "title": "Troubleshooting Registration and Sign-Up Issues",
        "summary": "Overcome common account creation problems, including verification link failures, duplicate email warnings, and password validation errors.",
        "steps": [
            "If you see an 'Email already registered' notice, use the 'Forgot Password' link at acme-store.example.com/login to recover your existing account.",
            "Ensure your password meets security standards: at least 8 characters, containing at least one uppercase letter, one number, and one special character.",
            "Clear your browser cookies and cache, or attempt registration in a private/incognito browsing tab if the submit button is unresponsive.",
            "Check your email spam folder if you have not received your account confirmation and activation email.",
            "If registration still fails, contact support@acme-store.example.com with the exact error message or screenshot for manual account provisioning.",
        ],
        "faqs": [
            (
                "Can two family members share one account email address?",
                "No. Each individual Acme Store account requires a unique email address to ensure account security and order privacy.",
            ),
            (
                "What should I do if my account activation link has expired?",
                "Attempt to log in with your email and password at acme-store.example.com/login to trigger an updated activation link to your inbox.",
            ),
        ],
    },
    "review": {
        "title": "How to Submit Product and Service Reviews",
        "summary": "Share your authentic feedback and product ratings to help fellow shoppers make informed purchasing decisions on Acme Store.",
        "steps": [
            "Log in to your account at acme-store.example.com and go to 'Order History'.",
            "Locate the delivered item you want to rate and click 'Write a Product Review'.",
            "Select an overall rating from 1 to 5 stars (with 5 stars being the highest).",
            "Write a descriptive review headline and detail your experience with product quality, fit, and performance.",
            "Optionally upload photos or videos of the product, then click 'Submit Review'.",
        ],
        "faqs": [
            (
                "How long does it take for a submitted review to appear publicly?",
                "Reviews undergo automated moderation to filter spam and offensive language and are published on the product page within 24 to 48 hours.",
            ),
            (
                "Can I edit or delete my review after publishing?",
                "Yes. Navigate to 'My Reviews' under your account profile at any time to revise your written review or change your star rating.",
            ),
        ],
    },
    "set_up_shipping_address": {
        "title": "How to Add and Manage Shipping Addresses",
        "summary": "Save multiple delivery addresses in your Acme Store address book for fast, convenient checkout for home, office, and gifts.",
        "steps": [
            "Log in to your account at acme-store.example.com.",
            "Click your profile icon in the top header and select 'My Account'.",
            "Select 'Shipping Addresses' or 'Address Book' from the account navigation menu.",
            "Click 'Add New Address' and enter the recipient's full name, street address, apartment/suite, city, state, zip code, and telephone number.",
            "Check the box marked 'Set as Default Address' if desired, then click 'Save Address'.",
        ],
        "faqs": [
            (
                "How many shipping addresses can I save in my address book?",
                "You can store up to 10 distinct shipping addresses in your Acme Store account for personal, business, and gift deliveries.",
            ),
            (
                "Does Acme Store deliver to PO Boxes and military APO/FPO addresses?",
                "Yes. We ship to standard PO Boxes and APO/FPO military addresses via standard USPS shipping methods.",
            ),
        ],
    },
    "switch_account": {
        "title": "How to Switch Between User Accounts",
        "summary": "Seamlessly toggle between personal, family, and business profiles on Acme Store without losing active browser sessions.",
        "steps": [
            "Locate the user profile icon in the upper-right corner of the acme-store.example.com navigation header.",
            "Click the user icon to open the account dropdown menu.",
            "Select 'Switch Account' from the list of options.",
            "Choose from your previously saved profiles or click 'Add Another Account'.",
            "Enter the login credentials for the secondary account and click 'Sign In' to switch active sessions.",
        ],
        "faqs": [
            (
                "Will switching accounts combine shopping carts or order histories?",
                "No. Each account retains an entirely separate shopping cart, saved payment methods, address book, and order history.",
            ),
            (
                "How do I remove an account profile from the quick-switch list?",
                "In the account menu, click 'Manage Accounts' and select 'Remove' next to the profile you wish to sign out and disconnect.",
            ),
        ],
    },
    "track_order": {
        "title": "How to Track Your Order and Shipment",
        "summary": "Track real-time shipment status, carrier milestones, and estimated delivery dates for all your Acme Store purchases.",
        "steps": [
            "Log in to your account at acme-store.example.com and go to 'Order History'.",
            "Find your order using the order number (#ACME-12345) and click 'Track Package'.",
            "View current shipment milestones: 'Order Placed', 'Shipped', 'In Transit', or 'Out for Delivery'.",
            "Click the carrier tracking number (UPS, FedEx, or USPS) to view detailed geolocation scans on the carrier portal.",
            "If checking out as a guest, visit acme-store.example.com/track and enter your order number and billing zip code.",
        ],
        "faqs": [
            (
                "When does package tracking information become active?",
                "Carrier tracking numbers activate within 24 hours of package handoff from our warehouse to the delivery carrier.",
            ),
            (
                "What does 'Out for Delivery' status mean?",
                "'Out for Delivery' indicates that the package is on the local delivery vehicle and will arrive at your address before 8:00 PM local time.",
            ),
        ],
    },
    "track_refund": {
        "title": "How to Track Your Refund Status",
        "summary": "Check the progress of returned items, monitor inspection approvals, and verify when refund credits will be posted to your payment method.",
        "steps": [
            "Sign in to your Acme Store account at acme-store.example.com.",
            "Navigate to 'Returns & Refunds' under your Account dashboard.",
            "Select your return claim using your return tracking number (e.g., #RET-77654) or original order number (#ACME-12345).",
            "Review the return progress: 'Package in Transit to Warehouse', 'Return Inspected & Approved', or 'Refund Issued'.",
            "Once status shows 'Refund Issued', check your bank or credit card statement within 3 to 5 business days for the posted credit.",
        ],
        "faqs": [
            (
                "How many days does it take for a refund to post to my original payment method?",
                "Once Acme Store approves and issues the refund, credit card companies and financial institutions typically post funds within 3 to 5 business days.",
            ),
            (
                "What happens if the original card used for payment has expired or been cancelled?",
                "Your financial institution will automatically reroute the refund to your replacement card or open account. Contact your bank if you no longer hold an account with them.",
            ),
        ],
    },
}


def build_knowledge_base() -> list[Path]:
    """Generate all 27 knowledge base articles from train split metadata."""
    df = prepare_dataset(DEFAULT_DATA_FILE)
    train_df = df[df["split"] == "train"]

    # Extract intent-to-category mapping from training data
    intent_categories = (
        train_df.groupby("intent")["category"].first().to_dict()
    )

    intents = sorted(intent_categories.keys())
    assert len(intents) == 27, f"Expected 27 intents, found {len(intents)}"

    KB_DIR.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    for idx, intent in enumerate(intents, start=1):
        stable_id = f"KB-{idx:02d}"
        category = intent_categories[intent]
        article_data = ARTICLES.get(intent)
        assert article_data is not None, f"Missing article content definition for {intent}"

        title = article_data["title"]
        summary = article_data["summary"]
        steps = article_data["steps"]
        faqs = article_data["faqs"]

        # Format markdown document
        md_lines = [
            "---",
            f"id: {stable_id}",
            f"intent: {intent}",
            f"category: {category}",
            "---",
            "",
            f"# {title}",
            "",
            summary,
            "",
            "## Steps",
            "",
        ]

        for step_num, step_text in enumerate(steps, start=1):
            md_lines.append(f"{step_num}. {step_text}")

        md_lines.extend([
            "",
            "## Frequently Asked Questions",
            "",
        ])

        for question, answer in faqs:
            md_lines.extend([
                f"### {question}",
                answer,
                "",
            ])

        content = "\n".join(md_lines).rstrip() + "\n"
        out_file = KB_DIR / f"{intent}.md"
        out_file.write_text(content, encoding="utf-8")
        generated_files.append(out_file)

    return generated_files


def main() -> None:
    files = build_knowledge_base()
    print(f"Successfully generated {len(files)} knowledge base articles in {KB_DIR}:\n")
    for f in files:
        print(f"  - {f.name}")
    print("\n--- Example Article: kb/cancel_order.md ---\n")
    example_path = KB_DIR / "cancel_order.md"
    print(example_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
