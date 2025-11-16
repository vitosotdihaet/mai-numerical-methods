test:
	cargo test $(TEST_SUITE) -- --nocapture --test-threads=1

test-plot:
	cargo test $(TEST_SUITE) --features plot_tests -- --nocapture --test-threads=1
