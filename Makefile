test:
	cargo test -- --nocapture --test-threads=1

test-plot:
	cargo test --features plot_tests -- --nocapture --test-threads=1
