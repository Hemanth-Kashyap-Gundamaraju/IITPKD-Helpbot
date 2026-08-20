def split_into_batches(items, batch_size):
    """
    Description: Splits a list into smaller lists (batches) of the given size.
        Pure logic, no side effects - shared by every ingestion path
        (Pinecone) so this logic only
        exists in one place.
    Inputs: items (list), batch_size (int). No globals read.
    Outputs: returns a list of lists. No globals changed.
    Dependencies: none.
    Utilities: available to Pinecone batch upload helpers.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
