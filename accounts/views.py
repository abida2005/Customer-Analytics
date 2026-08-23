from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db.models import Min
from collections import defaultdict
import pandas as pd
from .models import Shop, Transaction, UploadedDataset
from .utils import calculate_cohort, preprocess_dataset, calculate_rfm, calculate_churn
import hashlib
from functools import wraps
import re
import os
from django.shortcuts import get_object_or_404
from django.http import HttpResponseForbidden
from django.db.models import Min, Max
from datetime import datetime, date
from calendar import month_abbr

def shopkeeper_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect("login")
        if not hasattr(request.user, 'profile'):
            from .models import Profile
            Profile.objects.get_or_create(user=request.user, defaults={'role': 'SHOPKEEPER'})
        if request.user.profile.role in ['SHOPKEEPER', 'ADMIN']:
            return view_func(request, *args, **kwargs)
        return HttpResponseForbidden("Access Denied")
    return wrapper

# ---------------- REGISTER ----------------
def register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
            return render(request, "register.html")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return render(request, "register.html")

        pattern = r'^(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,}$'
        if not re.match(pattern, password1):
            messages.error(
                request,
                "Password must be at least 8 characters long, include one uppercase letter, one number, and one special character."
            )
            return render(request, "register.html")

        try:
            validate_password(password1, user=User(username=username))
        except ValidationError as e:
            for error in e.messages:
                messages.error(request, error)
            return render(request, "register.html")

        User.objects.create_user(username=username, email=email, password=password1)
        messages.success(request, "Registration successful.")
        user = authenticate(request, username=username, password=password1)
        if user:
            login(request, user)
            return redirect("shop_setup")
        return redirect("login")

    return render(request, "register.html")

# ---------------- SHOP SETUP ----------------
@login_required(login_url="/login/")
def shop_setup(request):
    if hasattr(request.user, 'shop') and request.user.shop.shop_name != 'My Shop':
        logout(request)
        request.session.flush()
        return redirect("login")

    if request.method == "POST":
        shop_name  = request.POST.get("shop_name", "").strip()
        owner_name = request.POST.get("owner_name", "").strip()
        phone      = request.POST.get("phone", "").strip()
        address    = request.POST.get("address", "").strip()

        if not shop_name or not owner_name:
            messages.error(request, "Shop Name and Owner Name are required.")
            return render(request, "shopsetup.html")

        shop, created = Shop.objects.get_or_create(
            user=request.user,
            defaults={'shop_name': shop_name, 'owner_name': owner_name, 'phone': phone, 'address': address}
        )
        if not created:
            shop.shop_name  = shop_name
            shop.owner_name = owner_name
            shop.phone      = phone
            shop.address    = address
            shop.save()

        logout(request)
        request.session.flush()
        messages.success(request, f"Shop '{shop_name}' set up! Please login to continue 🎉")
        return redirect("login")

    return render(request, "shopsetup.html")

# ---------------- LOGIN ----------------
def login_view(request):
    if request.user.is_authenticated:
        return redirect("profile")
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            if hasattr(user, 'profile') and user.profile.role == 'ADMIN':
                return redirect("/admin/")
            else:
                return redirect("profile")
        messages.error(request, "Invalid credentials")
    return render(request, "login.html")

def logout_view(request):
    logout(request)
    request.session.flush()
    return redirect("home")

# ---------------- PROFILE ----------------
@login_required(login_url="/login/")
@shopkeeper_required
def profile_view(request):
    shop, created = Shop.objects.get_or_create(
        user=request.user,
        defaults={'shop_name': 'My Shop', 'owner_name': request.user.username}
    )

    if request.method == "POST":
        shop_name  = request.POST.get("shop_name")
        owner_name = request.POST.get("owner_name")
        phone      = request.POST.get("phone", "").strip()
        address    = request.POST.get("address", "").strip()

        if shop_name and owner_name:
            shop.shop_name  = shop_name
            shop.owner_name = owner_name
            shop.phone      = phone
            shop.address    = address
            shop.save()
            messages.success(request, "Profile updated successfully! ✅")
        else:
            messages.error(request, "Please fill in all required fields.")
        return redirect("profile")

    datasets = UploadedDataset.objects.filter(user=request.user).order_by('-uploaded_at')
    dataset_count     = datasets.count()
    transaction_count = Transaction.objects.filter(user=request.user).count()
    customer_count    = Transaction.objects.filter(user=request.user).values('customer_id').distinct().count()

    context = {
        'shop': shop,
        'datasets': datasets,
        'dataset_count': dataset_count,
        'transaction_count': transaction_count,
        'customer_count': customer_count,
    }
    return render(request, "profile.html", context)

# ---------------- UPLOAD DATASET ----------------
@login_required(login_url="/login/")
@shopkeeper_required
def upload_dataset(request):
    f_invoice       = request.GET.get("f_invoice", "").strip()
    f_customer_id   = request.GET.get("f_customer_id", "").strip()
    f_customer_name = request.GET.get("f_customer_name", "").strip()
    f_bill          = request.GET.get("f_bill", "").strip()
    f_status        = request.GET.get("f_status", "").strip()
    f_year          = request.GET.get("f_year", "").strip()
    f_month         = request.GET.get("f_month", "").strip()
    f_day           = request.GET.get("f_day", "").strip()

    processed_data = Transaction.objects.filter(user=request.user).order_by("-id")

    if f_invoice:       processed_data = processed_data.filter(invoice_no__icontains=f_invoice)
    if f_customer_id:   processed_data = processed_data.filter(customer_id__icontains=f_customer_id)
    if f_customer_name: processed_data = processed_data.filter(customer_name__istartswith=f_customer_name)
    if f_year:          processed_data = processed_data.filter(transaction_date__year=f_year)
    if f_month:         processed_data = processed_data.filter(transaction_date__month=f_month)
    if f_day:           processed_data = processed_data.filter(transaction_date__day=f_day)
    if f_bill:          processed_data = processed_data.filter(bill_amount=f_bill)
    if f_status:        processed_data = processed_data.filter(status__iexact=f_status)

    is_filtered = any([f_invoice, f_customer_id, f_customer_name, f_year, f_month, f_day, f_bill, f_status])

    # CONFIRM REPLACE
    if request.method == "POST" and request.POST.get("confirm_replace") == "yes":
        file_path = request.session.get("pending_file_path")
        file_hash = request.session.get("pending_file_hash")
        if not file_path or not file_hash:
            messages.error(request, "Session expired. Please upload again.")
            return redirect("upload")
        df = preprocess_dataset(file_path)
        if df is None or df.empty:
            messages.error(request, "No valid rows found in CSV.")
            return redirect("upload")
        Transaction.objects.filter(user=request.user).delete()
        UploadedDataset.objects.filter(user=request.user).delete()
        transactions = [
            Transaction(
                user=request.user,
                invoice_no=row["InvoiceNo"],
                customer_id=row["CustomerID"],
                transaction_date=row["InvoiceDate"].date(),
                bill_amount=row["BillAmount"],
                customer_name=row.get("CustomerName", ""),
                status=row.get("Status", ""),
            )
            for _, row in df.iterrows()
        ]
        Transaction.objects.bulk_create(transactions)
        UploadedDataset.objects.create(user=request.user, file=file_path, file_hash=file_hash)
        request.session.pop("pending_file_path", None)
        request.session.pop("pending_file_hash", None)
        messages.success(request, "Dataset replaced successfully ✅")
        return redirect("upload")

    # MANUAL ADD
    if request.method == "POST" and request.POST.get("manual_add") == "yes":
        invoice_no       = request.POST.get("invoice_no")
        customer_id      = request.POST.get("customer_id")
        transaction_date = request.POST.get("transaction_date")
        bill_amount      = request.POST.get("bill_amount")
        if invoice_no and customer_id and transaction_date and bill_amount:
            Transaction.objects.create(
                user=request.user,
                invoice_no=invoice_no,
                customer_id=customer_id,
                transaction_date=transaction_date,
                bill_amount=bill_amount,
                customer_name=request.POST.get("customer_name", ""),
                status=request.POST.get("status", ""),
            )
            messages.success(request, "Transaction added successfully 🎉")
        else:
            messages.error(request, "Please fill required fields.")
        return redirect("upload")

    # NORMAL UPLOAD
    if request.method == "POST" and request.FILES.get("dataset"):
        csv_file   = request.FILES["dataset"]
        file_bytes = csv_file.read()
        file_hash  = hashlib.sha256(file_bytes).hexdigest()
        csv_file.seek(0)

        if UploadedDataset.objects.filter(user=request.user, file_hash=file_hash).exists():
            file_path = default_storage.save(csv_file.name, csv_file)
            request.session["pending_file_path"] = file_path
            request.session["pending_file_hash"] = file_hash
            request.session["duplicate_detected"] = True
            return redirect("upload")

        file_path = default_storage.save(csv_file.name, csv_file)
        df = preprocess_dataset(file_path)
        if df is None or df.empty:
            messages.error(request, "No valid rows found in CSV.")
            return redirect("upload")
        Transaction.objects.filter(user=request.user).delete()
        dataset_record = UploadedDataset.objects.create(user=request.user, file=file_path, file_hash=file_hash)
        Transaction.objects.bulk_create([
            Transaction(
                user=request.user,
                invoice_no=row["InvoiceNo"],
                customer_id=row["CustomerID"],
                transaction_date=row["InvoiceDate"].date(),
                bill_amount=row["BillAmount"],
                customer_name=row.get("CustomerName", ""),
                status=row.get("Status", ""),
                dataset=dataset_record,
            )
            for _, row in df.iterrows()
        ])
        messages.success(request, "Dataset uploaded successfully 🎉")
        return redirect("upload")

    duplicate_detected = request.session.pop("duplicate_detected", False)
    all_transactions   = Transaction.objects.filter(user=request.user)
    total_records      = all_transactions.count()
    date_range         = all_transactions.aggregate(start=Min("transaction_date"), end=Max("transaction_date")) \
                         if total_records > 0 else {"start": None, "end": None}

    return render(request, "upload.html", {
        "processed_data"    : processed_data,
        "duplicate_detected": duplicate_detected,
        "f_invoice"         : f_invoice,
        "f_customer_id"     : f_customer_id,
        "f_customer_name"   : f_customer_name,
        "f_year"            : f_year,
        "f_month"           : f_month,
        "f_day"             : f_day,
        "f_bill"            : f_bill,
        "f_status"          : f_status,
        "is_filtered"       : is_filtered,
        "total_records"     : total_records,
        "date_start"        : date_range["start"],
        "date_end"          : date_range["end"],
    })


@login_required(login_url="/login/")
@shopkeeper_required
def delete_dataset(request, pk):
    dataset = get_object_or_404(UploadedDataset, pk=pk, user=request.user)
    if request.method == "POST":
        Transaction.objects.filter(dataset=dataset).delete()
        Transaction.objects.filter(user=request.user, dataset__isnull=True).delete()
        if dataset.file:
            try:
                file_path = dataset.file.path
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"File deletion skipped: {e}")
        dataset.delete()
        messages.success(request, "Dataset and all related transactions deleted successfully 🗑️")
    return redirect("profile")


# ---------------- INSIGHTS (RFM + CHURN + COHORT) ----------------
# ─────────────────────────────────────────────────────────────────────────────
# REPLACE your existing insights_view with this version.
# Only this view changes — everything else in views.py stays identical.
# ─────────────────────────────────────────────────────────────────────────────
# CHANGES vs original:
#  1. calculate_cohort() now returns (cohort_df, forecast_info) — unpacked here
#  2. churn_sorted now includes ChurnProbability column (new ML field)
#  3. cohort forecast data passed to template context
# ─────────────────────────────────────────────────────────────────────────────

@login_required(login_url="/login/")
@shopkeeper_required
def insights_view(request):
    rfm_df   = calculate_rfm(request.user)
    churn_df = calculate_churn(request.user)

    cohort_result = calculate_cohort(request.user)
    if isinstance(cohort_result, tuple):
        cohort_df_raw, cohort_forecast = cohort_result
    else:
        cohort_df_raw, cohort_forecast = cohort_result, {}

    if rfm_df is None or rfm_df.empty:
        return render(request, "insights.html", {"no_data": True})

    # ── RFM ──────────────────────────────────────────────────────────────────
    segment_counts   = rfm_df["Segment"].value_counts().to_dict()
    selected_segment = request.GET.get("segment", "All")

    # FIX: filter uses the EXACT segment labels produced by calculate_rfm()
    if selected_segment == "All":
        filtered_rfm_df = rfm_df
    else:
        filtered_rfm_df = rfm_df[rfm_df["Segment"] == selected_segment]

    rfm_sorted = filtered_rfm_df.sort_values(by="Monetary", ascending=False)

    # ── Churn ─────────────────────────────────────────────────────────────────
    churn_counts  = churn_df["ChurnRisk"].value_counts().to_dict()
    selected_risk = request.GET.get("risk", "All")

    # FIX: filter uses EXACT ChurnRisk labels from calculate_churn()
    if selected_risk == "All":
        filtered_churn_df = churn_df
    elif selected_risk == "Leaving Very Soon":
        filtered_churn_df = churn_df[churn_df["ChurnRisk"] == "Leaving Very Soon"]
    elif selected_risk == "Needs Your Attention":
        filtered_churn_df = churn_df[churn_df["ChurnRisk"] == "Needs Your Attention"]
    elif selected_risk == "Still Active":
        filtered_churn_df = churn_df[churn_df["ChurnRisk"] == "Still Active"]
    else:
        filtered_churn_df = churn_df

    churn_sorted = filtered_churn_df.sort_values(by="DaysSinceLastPurchase", ascending=False)

    # ── Cohort Analysis ───────────────────────────────────────────────────────
    cohort_table  = []
    chart_totals  = defaultdict(int)
    cohort_message= ""
    cohort_data   = {}

    if not cohort_df_raw.empty:
        transactions = Transaction.objects.filter(user=request.user)
        first_purchase     = transactions.values('customer_id').annotate(first_date=Min('transaction_date'))
        first_purchase_map = {item['customer_id']: item['first_date'] for item in first_purchase}
        cohort_data        = defaultdict(lambda: defaultdict(set))
        for t in transactions:
            first_date   = first_purchase_map[t.customer_id]
            cohort_month = first_date.strftime('%Y-%m')
            month_diff   = (t.transaction_date.year  - first_date.year) * 12 + \
                           (t.transaction_date.month - first_date.month)
            cohort_data[cohort_month][month_diff].add(t.customer_id)

        for cohort_month in sorted(cohort_data.keys()):
            row = {'cohort': cohort_month}
            for m in range(5):
                count = len(cohort_data[cohort_month].get(m, []))
                row[f'month_{m}'] = count
                chart_totals[m]  += count
            cohort_table.append(row)
    else:
        cohort_message = "Upload more historical data to view cohort trends."

    # ── Dynamic month labels ──────────────────────────────────────────────────
    def build_month_labels(cohort_data_dict):
        sorted_months = sorted(cohort_data_dict.keys()) if cohort_data_dict else []
        if not sorted_months:
            return ["Same Month", "After 1 Month", "After 2 Months", "After 3 Months", "After 4 Months"]
        base_year, base_month = map(int, sorted_months[0].split('-'))
        labels = []
        for i in range(5):
            y          = base_year + (base_month - 1 + i) // 12
            m          = (base_month - 1 + i) % 12 + 1
            month_name = datetime(y, m, 1).strftime('%b %Y')
            if i == 0:
                labels.append(f"Same Month ({month_name})")
            else:
                suffix = "Month" if i == 1 else "Months"
                labels.append(f"After {i} {suffix} ({month_name})")
        return labels

    cohort_graph_labels = build_month_labels(cohort_data)
    cohort_graph_values = [chart_totals[i] for i in range(5)]

    # ── Cohort insights ───────────────────────────────────────────────────────
    total_customers         = sum(chart_totals.values())
    returned_after_1_month  = chart_totals.get(1, 0)
    drop_percentage         = round(
        ((total_customers - returned_after_1_month) / total_customers) * 100, 1
    ) if total_customers > 0 else 0

    cohort_insights = {
        "drop_percentage":        drop_percentage,
        "returned_after_1_month": returned_after_1_month,
        "ml_trend":               cohort_forecast.get("trend", "N/A"),
        "ml_slope":               cohort_forecast.get("slope", 0),
        "ml_forecast_offsets":    cohort_forecast.get("forecast_month_offsets", []),
        "ml_forecast_values":     cohort_forecast.get("forecast_values", []),
        "ml_retention_rates":     cohort_forecast.get("retention_rates", {}),
    }

    # ── Context ───────────────────────────────────────────────────────────────
    context = {
        "no_data": False,
        # RFM — using exact ML segment labels
        "champions":        segment_counts.get("Best Customers",    0),
        "loyal":            segment_counts.get("Regular Customers", 0),
        "at_risk":          segment_counts.get("Slipping Away",     0),
        "lost":             segment_counts.get("Lost Customers",    0),
        "selected_segment": selected_segment,
        "rfm_page_obj":     rfm_sorted.to_dict(orient="records"),
        # Churn — FIX: use exact ChurnRisk labels
        "high_churn":       churn_counts.get("Leaving Very Soon",    0),
        "medium_churn":     churn_counts.get("Needs Your Attention", 0),
        "low_churn":        churn_counts.get("Still Active",         0),  # FIX was "Low Risk"
        "selected_risk":    selected_risk,
        "churn_page_obj":   churn_sorted.to_dict(orient="records"),
        # Cohort
        "cohort_table":         cohort_table,
        "cohort_message":       cohort_message,
        "cohort_graph_labels":  cohort_graph_labels,
        "cohort_graph_values":  cohort_graph_values,
        "cohort_insights":      cohort_insights,
        # FIX: pass active tab back so JS can restore the right section after filter
        "active_tab":           request.GET.get("tab", "customers"),
    }
    return render(request, "insights.html", context)
# ---------------- DELETE TRANSACTION ----------------
@login_required(login_url="/login/")
@shopkeeper_required
def delete_transaction(request, pk):
    transaction = get_object_or_404(Transaction, pk=pk, user=request.user)
    transaction.delete()
    return redirect("upload")


# ---------------- EDIT TRANSACTION ----------------
@login_required(login_url="/login/")
@shopkeeper_required
def edit_transaction(request, pk):
    transaction = get_object_or_404(Transaction, pk=pk, user=request.user)
    if request.method == "POST":
        transaction.invoice_no       = request.POST.get("invoice_no")
        transaction.customer_id      = request.POST.get("customer_id")
        transaction.transaction_date = request.POST.get("transaction_date")
        transaction.bill_amount      = request.POST.get("bill_amount")
        transaction.customer_name    = request.POST.get("customer_name", "")
        transaction.status           = request.POST.get("status", "")
        transaction.save()
        return redirect("upload")
    return render(request, "edit_transaction.html", {"transaction": transaction})