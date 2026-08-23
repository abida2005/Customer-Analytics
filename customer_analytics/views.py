from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import FileResponse, Http404
import os
from accounts.models import Transaction, Report
from accounts.utils import (
    calculate_rfm,
    calculate_churn,
    calculate_cohort,
    generate_rfm_report,
    generate_churn_report,
    generate_cohort_report,
)


def home(request):
    if request.user.is_authenticated:
        return redirect("profile")
    return render(request, "index.html")


@login_required(login_url="/login/")
def download_rfm_report(request):
    rfm_df = calculate_rfm(request.user)

    if rfm_df is None or rfm_df.empty:
        messages.error(request, "No data available for RFM report")
        return redirect("insights")

    file_path = generate_rfm_report(request.user, rfm_df)

    if not os.path.exists(file_path):
        raise Http404("Report file not found.")

    response = FileResponse(open(file_path, "rb"), content_type="application/pdf")
    response["Content-Disposition"] = 'inline; filename="RFM_Report.pdf"'
    return response


@login_required(login_url="/login/")
def download_churn_report(request):
    churn_df = calculate_churn(request.user)

    if churn_df is None or churn_df.empty:
        messages.error(request, "No data available for churn report")
        return redirect("insights")

    file_path = generate_churn_report(request.user, churn_df)

    if not os.path.exists(file_path):
        raise Http404("Report file not found.")

    response = FileResponse(open(file_path, "rb"), content_type="application/pdf")
    response["Content-Disposition"] = 'inline; filename="Churn_Report.pdf"'
    return response


@login_required(login_url="/login/")
def download_cohort_report(request):
    cohort_result = calculate_cohort(request.user)

    # calculate_cohort() always returns a (df, forecast_info) tuple
    if isinstance(cohort_result, tuple):
        cohort_df, forecast_info = cohort_result
    else:
        cohort_df, forecast_info = cohort_result, {}

    if cohort_df is None or cohort_df.empty:
        messages.error(request, "No cohort data available. Upload data from at least 2 months.")
        return redirect("insights")

    # Pass tuple so generate_cohort_report can use both df + forecast
    file_path = generate_cohort_report(request.user, (cohort_df, forecast_info))

    if not os.path.exists(file_path):
        raise Http404("Report file not found.")

    response = FileResponse(open(file_path, "rb"), content_type="application/pdf")
    response["Content-Disposition"] = 'inline; filename="Cohort_Report.pdf"'
    return response