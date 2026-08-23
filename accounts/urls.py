
from django.urls import path
from .views import login_view,shop_setup, profile_view,delete_dataset, logout_view, register, upload_dataset,insights_view, delete_transaction, edit_transaction

urlpatterns = [
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("register/", register, name="register"),
    path("profile/", profile_view, name="profile"),  # New profile route (replaces dashboard)
    path("upload/", upload_dataset, name="upload"),
    path("insights/", insights_view, name="insights"),
    path("transaction/delete/<int:pk>/", delete_transaction, name="delete_transaction"),
    path("transaction/edit/<int:pk>/", edit_transaction, name="edit_transaction"),
    path('delete-dataset/<int:pk>/', delete_dataset, name='delete_dataset'),
# In your accounts/urls.py, add this path:
    path('shop-setup/', shop_setup, name='shop_setup'),
]


