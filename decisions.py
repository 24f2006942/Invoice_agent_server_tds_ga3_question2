            f"AI provider request timed out ({type(exc).__name__})."
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"AI provider network request failed ({type(exc).__name__})."
        ) from exc

    except ValueError as exc:
        raise RuntimeError("AI provider returned an invalid response.") from exc

    try:
        content = response_data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        content = None

    if not content:
        raise RuntimeError("AI returned an empty response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AI returned invalid JSON.") from exc

    proposals = parsed.get("proposals") if isinstance(parsed, dict) else None

    if not isinstance(proposals, list):
        raise RuntimeError("AI response must contain a proposals list.")

    expected_ids = {package_id for package_id, _ in packages}
    result = {}

    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise RuntimeError("AI returned a malformed proposal.")

        package_id = proposal.get("packageId")

        if not isinstance(package_id, str) or package_id not in expected_ids:
            raise RuntimeError("AI returned an unknown packageId.")

        if package_id in result:
            raise RuntimeError("AI returned a duplicate packageId.")

        result[package_id] = proposal

    if set(result) != expected_ids:
        raise RuntimeError("AI did not return exactly one proposal per package.")

    return result


async def decide_packages(
    packages: list[dict[str, Any]],
    policy_revision: str = "",
) -> list[dict[str, Any]]:
    package_ids = [package_id_of(p) for p in packages]

    if len(package_ids) != len(set(package_ids)):
        raise ValueError("Package IDs must be unique.")

    keys = [
        package_cache_key(package, policy_revision)
        for package in packages
    ]

    cached = get_cached_decisions(keys)
    misses = []

    for package_id, package, key in zip(package_ids, packages, keys):
        if key not in cached:
            misses.append((package_id, package, key))

    newly_validated = {}
    validation_errors = []

    # All uncached packages are sent in one AI request.
    if misses:
        model_results = await call_model(
            [(package_id, package) for package_id, package, _ in misses],
            policy_revision,
        )

        for package_id, package, key in misses:
            try:
                validated = validate_decision(
                    model_results[package_id],
                    package_id,
                    package,
                )
                newly_validated[key] = validated
            except ValueError as exc:
                validation_errors.append(str(exc))

        save_cached_decisions(newly_validated)
        cached.update(newly_validated)

        if validation_errors:
            details = "; ".join(validation_errors[:3])
            raise ValueError(
                "One or more AI decisions failed validation: " + details
            )

    proposals = []

    for package_id, package, key in zip(package_ids, packages, keys):
        decision = validate_decision(
            cached[key],
            package_id,
            package,
        )

        proposals.append(
            {
                "packageId": package_id,
                "actionId": "act_" + uuid.uuid4().hex,
                **decision,
            }
        )

    if len({p["actionId"] for p in proposals}) != len(proposals):
        raise RuntimeError("Action IDs must be unique.")

    return proposals