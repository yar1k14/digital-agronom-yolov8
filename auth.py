# -*- coding: utf-8 -*-
"""
auth.py — Модуль авторизации для приложения сортировки помидоров.
Хранит пользователей в файле users.json (хэши паролей sha256).
"""

import json
import hashlib
import os
import re
from datetime import datetime

USERS_FILE = "users.json"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def register_user(username: str, password: str, confirm: str, full_name: str = "") -> tuple[bool, str]:
    username = username.strip()
    full_name = full_name.strip()

    if len(username) < 3:
        return False, "Логин должен содержать минимум 3 символа"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Логин может содержать только латинские буквы, цифры и _"
    if len(password) < 6:
        return False, "Пароль должен содержать минимум 6 символов"
    if password != confirm:
        return False, "Пароли не совпадают"

    users = _load_users()
    if username.lower() in {u.lower() for u in users}:
        return False, "Пользователь с таким логином уже существует"

    users[username] = {
        "password_hash": _hash(password),
        "full_name": full_name or username,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": None,
    }
    _save_users(users)
    return True, "Регистрация прошла успешно!"


def login_user(username: str, password: str) -> tuple[bool, str, dict]:
    username = username.strip()
    if not username or not password:
        return False, "Заполните все поля", {}

    users = _load_users()
    # Регистронезависимый поиск
    matched_key = next((k for k in users if k.lower() == username.lower()), None)
    if matched_key is None:
        return False, "Неверный логин или пароль", {}

    user = users[matched_key]
    if user["password_hash"] != _hash(password):
        return False, "Неверный логин или пароль", {}

    # Обновляем время последнего входа
    users[matched_key]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_users(users)

    return True, "Добро пожаловать!", {
        "username": matched_key,
        "full_name": user.get("full_name", matched_key),
        "created_at": user.get("created_at", ""),
        "last_login": user.get("last_login", ""),
    }
