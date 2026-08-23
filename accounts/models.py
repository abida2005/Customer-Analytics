import os
from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save

class Profile(models.Model):
    ROLE_CHOICES = (
        ('ADMIN', 'Admin'),
        ('SHOPKEEPER', 'Shopkeeper'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='SHOPKEEPER')
    shop_name = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return self.user.username

class Shop(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    shop_name = models.CharField(max_length=200)
    owner_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    phone = models.CharField(max_length=20, blank=True, default='')
    address = models.TextField(blank=True, default='')

    def __str__(self):
        return self.shop_name
    
class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    invoice_no = models.CharField(max_length=50)
    customer_id = models.CharField(max_length=50)
    transaction_date = models.DateField()
    customer_name = models.CharField(max_length=100, blank=True, default='')
    status = models.CharField(max_length=20, blank=True, default='')


    bill_amount = models.FloatField()
    dataset = models.ForeignKey(
        "UploadedDataset",
        on_delete=models.CASCADE,
        related_name="transactions",
        null=True,
        blank=True
    )

    def __str__(self):
        return f"{self.invoice_no} - {self.customer_id}"

class Report(models.Model):
    REPORT_CHOICES = (
        ('RFM', 'RFM Analysis'),
        ('CHURN', 'Churn Analysis'),
        ('COHORT', 'Cohort Analysis'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    report_type = models.CharField(max_length=20, choices=REPORT_CHOICES)
    file = models.FileField(upload_to='reports/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.report_type} - {self.user.username}"

from django.contrib.auth.models import User
from django.db import models

class UploadedDataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='datasets/')
    file_hash = models.CharField(max_length=64 , default="", blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.file.name}"

    # ✅ Safe file size — won't crash if file is missing from disk
    @property
    def safe_file_size(self):
        try:
            if self.file and os.path.isfile(self.file.path):
                size = os.path.getsize(self.file.path)
                # Convert to human readable
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size < 1024:
                        return f"{size:.1f} {unit}"
                    size /= 1024
                return f"{size:.1f} GB"
        except Exception:
            pass
        return "N/A"

    # ✅ Safe file name — strips the 'datasets/' folder prefix
    @property
    def file_display_name(self):
        if self.file:
            return os.path.basename(self.file.name)
        return "Unknown"