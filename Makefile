test:
	cargo test $(TEST_SUITE) -- --nocapture

test-plot:
	cargo test $(TEST_SUITE) --features plot_tests -- --nocapture
