"""Tests du modèle « livres » (PR 2-a) : mode, invariants de configuration, quantités, réconciliation."""

import pytest

from src import portfolio_books as pb


def test_mode_defaults_to_legacy(monkeypatch):
    monkeypatch.delenv("STRATEGY_MODE", raising=False)
    assert pb.get_strategy_mode() == "legacy"
    monkeypatch.setenv("STRATEGY_MODE", "")
    assert pb.get_strategy_mode() == "legacy"


def test_mode_core_sleeve_and_unknown(monkeypatch):
    monkeypatch.setenv("STRATEGY_MODE", " Core_Sleeve ")
    assert pb.get_strategy_mode() == "core_sleeve"
    monkeypatch.setenv("STRATEGY_MODE", "turbo")
    with pytest.raises(ValueError):
        pb.get_strategy_mode()


def test_default_books_are_valid_and_split_90_10():
    books = pb.default_books()
    pb.validate_books(books)
    assert books["core"].target_weight == pytest.approx(0.90)
    assert sum(b.target_weight for b in books.values() if b.is_active) == pytest.approx(0.10)
    assert books["core"].protection == "none"


def test_oil_counts_inside_the_ten_percent():
    books = pb.default_books()
    books["sleeve"] = pb.Book("sleeve", "sleeve", 0.06, ("SXRVd_EQ",))
    books["oil"] = pb.Book("oil", "oil", 0.04, ("OD7Fd_EQ",))
    books["core"] = pb.Book("core", "core", 0.90, ("SXRVd_EQ",))
    pb.validate_books(books)
    books["core"] = pb.Book("core", "core", 0.85, ("SXRVd_EQ",))
    books["oil"] = pb.Book("oil", "oil", 0.05, ("OD7Fd_EQ",))  # 6 % + 5 % = 11 % > plafond
    with pytest.raises(ValueError, match="plafond"):
        pb.validate_books(books)


def test_core_cannot_have_a_broker_stop():
    books = pb.default_books()
    books["core"] = pb.Book("core", "core", 0.90, ("SXRVd_EQ",), protection="gtc_stop")
    with pytest.raises(ValueError, match="cœur"):
        pb.validate_books(books)


@pytest.mark.parametrize("bad", [
    pb.Book("x", "core", -0.1, ("A",)),
    pb.Book("x", "wizard", 0.1, ("A",)),
    pb.Book("x", "sleeve", 0.1, ()),
    pb.Book("x", "sleeve", 0.1, ("A",), protection="magic"),
])
def test_invalid_books_rejected(bad):
    with pytest.raises(ValueError):
        pb.validate_books({"x": bad})


def test_total_weight_over_100_rejected():
    with pytest.raises(ValueError, match="100"):
        pb.validate_books({"a": pb.Book("a", "core", 0.7, ("A",)), "b": pb.Book("b", "core", 0.4, ("B",))})


def test_target_amounts_for_30k():
    assert pb.target_amounts(pb.default_books(), 30000.0) == {"core": 27000.0, "sleeve": 3000.0, "oil": 0.0}
    with pytest.raises(ValueError):
        pb.target_amounts(pb.default_books(), -1.0)


def test_same_instrument_shared_between_books_and_reconcile():
    st = pb.BooksState()
    st.set_quantity("core", "SXRVd_EQ", 20.0)
    st.set_quantity("sleeve", "SXRVd_EQ", 2.0)
    assert st.total_quantity("SXRVd_EQ") == 22.0
    assert pb.reconcile(st, {"SXRVd_EQ": 22.0}) == {}
    assert pb.reconcile(st, {"SXRVd_EQ": 22.0000001}) == {}  # arrondi broker toléré
    assert pb.reconcile(st, {"SXRVd_EQ": 23.5}) == {"SXRVd_EQ": pytest.approx(1.5)}
    # instrument inconnu de l'état (achat manuel) et instrument disparu du broker
    drift = pb.reconcile(st, {"SXRVd_EQ": 22.0, "OD7Fd_EQ": 3.0})
    assert drift == {"OD7Fd_EQ": 3.0}
    assert pb.reconcile(st, {}) == {"SXRVd_EQ": -22.0}


def test_negative_quantity_rejected():
    with pytest.raises(ValueError):
        pb.BooksState().set_quantity("core", "SXRVd_EQ", -1.0)


def test_peak_only_rises_and_drawdown():
    st = pb.BooksState()
    assert st.update_peak(30000.0) == 0.0
    assert st.update_peak(33000.0) == 0.0
    assert st.update_peak(21450.0) == pytest.approx(-0.35)
    assert st.equity_peak == 33000.0
    assert pb.BooksState().update_peak(0.0) == 0.0


def test_state_roundtrip_atomic(tmp_path):
    path = tmp_path / "books.json"
    assert pb.load_state(path).quantities == {}
    st = pb.BooksState(equity_peak=31000.0)
    st.set_quantity("core", "SXRVd_EQ", 20.5)
    pb.save_state(st, path)
    back = pb.load_state(path)
    assert back.quantity("core", "SXRVd_EQ") == 20.5 and back.equity_peak == 31000.0
    assert list(tmp_path.glob("*.tmp")) == []
