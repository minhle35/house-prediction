from datetime import date

from pydantic import BaseModel


class PropertyFeatures(BaseModel):
    """Raw input features for a single property — mirrors the training CSV schema."""

    id: str | int | None = None
    region: str
    suburb: str | None = None
    postcode: str | int | None = None
    date_sold: date  # validated as YYYY-MM-DD; rejects "string", "tomorrow", etc.

    # Property
    # Note: type and price are each other's cross-target features due to how the
    # models were trained (split_x_y only drops the target + id, not the other target).
    # type  → required by the regression pipeline as a feature
    # price → required by the classification pipeline as a feature
    type: str | None = None
    price: float | None = None
    num_bed: float | None = None
    num_bath: float | None = None
    num_parking: float | None = None
    property_size: float | None = None
    property_inflation_index: float | None = None

    # Suburb / location
    suburb_population: float | None = None
    suburb_median_income: float | None = None
    suburb_sqkm: float | None = None
    suburb_lat: float | None = None
    suburb_lng: float | None = None
    suburb_elevation: float | None = None
    km_from_cbd: float | None = None
    suburb_median_house_price: float | None = None
    suburb_median_apartment_price: float | None = None

    # Rental market
    median_house_rent_per_week: float | None = None
    median_apartment_rent_per_week: float | None = None
    avg_years_held: float | None = None

    # Transport
    time_to_cbd_public_transport_town_hall_st: float | None = None
    time_to_cbd_driving_town_hall_st: float | None = None

    # Suburb demographics
    suburbpopulation: float | None = None
    public_housing_pct: float | None = None
    ethnic_breakdown: str | None = None

    # Suburb character
    highlights_attractions: str | None = None
    ideal_for: str | None = None
    nearest_train_station: str | None = None

    # Economic
    cash_rate: float | None = None

    # Livability scores
    traffic: float | None = None
    public_transport: float | None = None
    affordability_rental: float | None = None
    affordability_buying: float | None = None
    nature: float | None = None
    noise: float | None = None
    things_to_see_do: float | None = None
    family_friendliness: float | None = None
    pet_friendliness: float | None = None
    safety: float | None = None
    overall_rating: float | None = None


class PredictRequest(BaseModel):
    data: list[PropertyFeatures]
